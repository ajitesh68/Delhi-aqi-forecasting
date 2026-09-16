"""Email capture and the record of who has used this."""

import datetime as dt
import re

import streamlit as st

SHEET_HEADERS = ["email", "first_seen", "last_seen", "visits"]
WORKSHEET = "users"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def sheet_configured():
    try:
        return bool(st.secrets.get("gcp_service_account")) and bool(
            st.secrets.get("users_sheet_id"))
    except Exception:
        return False


def current_user():
    email = st.session_state.get("user_email")
    return {"email": email} if email else None


def require_login():
    """Gate the page. Returns the signed-in user or stops the script."""
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
        'forecast. Enter your email to continue.</div>', unsafe_allow_html=True)
    st.markdown("")
    email = st.text_input("Email address", placeholder="you@example.com")
    if st.button("Continue", type="primary"):
        clean = email.strip()
        if not EMAIL_RE.match(clean):
            st.error("That doesn't look like a valid email address.")
        else:
            st.session_state["user_email"] = clean
            st.rerun()
    st.caption("Your email address is recorded so the number of people "
               "using this can be counted. Nothing else is asked for, and "
               "it is never shared or sold. This is not verified against "
               "anything -- it is a count, not an account.")


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
        sheet.update("A1", [SHEET_HEADERS])
    return sheet


def _upsert(user):
    sheet = _worksheet()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cell = sheet.find(user["email"], in_column=1)
    if cell is None:
        sheet.append_row([user["email"], now, now, 1])
        return
    row = sheet.row_values(cell.row)
    visits = int(row[3]) + 1 if len(row) > 3 and str(row[3]).isdigit() else 1
    sheet.update(f"C{cell.row}:D{cell.row}", [[now, visits]])


def sidebar_account(user):
    st.markdown("---")
    st.caption(f"Using as **{user['email']}**")
    if st.button("Sign out", width="stretch"):
        del st.session_state["user_email"]
        st.rerun()
