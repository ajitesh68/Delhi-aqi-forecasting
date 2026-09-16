"""Google sign-in and the record of who has signed in."""

import datetime as dt

import streamlit as st

SHEET_HEADERS = ["email", "name", "first_seen", "last_seen", "visits"]
WORKSHEET = "users"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def auth_configured():
    try:
        return bool(st.secrets.get("auth", {}).get("client_id"))
    except Exception:
        return False


def sheet_configured():
    try:
        return bool(st.secrets.get("gcp_service_account")) and bool(
            st.secrets.get("users_sheet_id"))
    except Exception:
        return False


def current_user():
    if not auth_configured() or not st.user.get("is_logged_in"):
        return None
    return {"email": st.user.get("email"),
            "name": st.user.get("name") or st.user.get("email"),
            "picture": st.user.get("picture")}


def require_login():
    """Gate the page. Returns the signed-in user or stops the script."""
    if not auth_configured():
        st.error("Sign-in is not configured on this deployment. "
                 "An `[auth]` section is missing from Streamlit secrets.")
        st.stop()

    user = current_user()
    if user is None:
        _login_screen()
        st.stop()

    _record_once(user)
    return user


def _login_screen():
    st.markdown("# Delhi Air")
    st.markdown(
        '<div class="note">Live air quality from the CPCB / DPCC '
        'ground-monitoring network across Delhi, with a 24-hour PM2.5 '
        'forecast. Sign in to continue.</div>', unsafe_allow_html=True)
    st.markdown("")
    if st.button("Sign in with Google", type="primary"):
        st.login()
    st.caption("Your name and email address are recorded so the number of "
               "people using this can be counted. Nothing else is stored, "
               "and it is never shared or sold. Google handles the password; "
               "this app never sees it.")


def _record_once(user):
    key = f"recorded:{user['email']}"
    if st.session_state.get(key) or not sheet_configured():
        return
    st.session_state[key] = True
    try:
        _upsert(user)
    except Exception as exc:
        st.session_state[f"record_error:{user['email']}"] = str(exc)


@st.cache_resource(show_spinner=False)
def _worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
    book = gspread.authorize(creds).open_by_key(st.secrets["users_sheet_id"])
    try:
        sheet = book.worksheet(WORKSHEET)
    except Exception:
        sheet = book.add_worksheet(WORKSHEET, rows=1000, cols=len(SHEET_HEADERS))
        sheet.append_row(SHEET_HEADERS)
    if sheet.row_values(1) != SHEET_HEADERS:
        sheet.insert_row(SHEET_HEADERS, 1)
    return sheet


def _upsert(user):
    sheet = _worksheet()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cell = sheet.find(user["email"], in_column=1)
    if cell is None:
        sheet.append_row([user["email"], user["name"], now, now, 1])
        return
    row = sheet.row_values(cell.row)
    visits = int(row[4]) + 1 if len(row) > 4 and str(row[4]).isdigit() else 1
    sheet.update(f"D{cell.row}:E{cell.row}", [[now, visits]])


def sidebar_account(user):
    st.markdown("---")
    st.caption(f"Signed in as **{user['name']}**")
    if st.button("Sign out", width="stretch"):
        st.logout()
