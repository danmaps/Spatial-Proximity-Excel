"""
gisapps_header.py — Shared header bar for Streamlit-based GIS Apps.

Python equivalent of the gisapps-header.js injected script.
Streamlit controls its own DOM, so the static-HTML <script> approach
doesn't work.  This module uses st.markdown(unsafe_allow_html=True)
to render a matching header bar at the top of the page.

Usage (add near the top of your Streamlit app, after st.set_page_config):

    from gisapps_header import gisapps_header
    gisapps_header()
"""

import os
import streamlit as st


def gisapps_header(
    title: str | None = None,
    portal_port: int = 8000,
):
    """Render the GIS Apps header bar inside a Streamlit app.

    Parameters
    ----------
    title : str, optional
        Text shown in the header.  Defaults to the page_title from
        set_page_config (pulled from st.get_option) or "GIS App".
    portal_port : int
        Port where the GIS Apps Portal is running (default 8000).
    """
    if title is None:
        try:
            title = st.get_option("browser.tabTitle") or "GIS App"
        except Exception:
            title = "GIS App"

    portal_url = f"//localhost:{portal_port}"

    st.markdown(
        f"""
        <style>
        /* ── gisapps header ─────────────────────────────────────── */
        #gisapps-header {{
            height: 48px; background: #1a1a2e; color: #fff;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 16px; z-index: 1000; flex-shrink: 0;
            font-family: "Avenir Next","Segoe UI",sans-serif;
            position: fixed; top: 0; left: 0; width: 100%;
            box-sizing: border-box;
        }}
        #gisapps-header h1 {{
            font-size: 1.1rem; font-weight: 600; letter-spacing: .3px;
            margin: 0; color: #fff;
        }}
        #gisapps-header a {{
            color: #aaa; font-size: .8rem; text-decoration: none;
        }}
        #gisapps-header a:hover {{ color: #fff; }}

        /* Push Streamlit content below the fixed header */
        .stApp > header {{ top: 48px !important; }}
        .stMainBlockContainer {{ padding-top: 64px !important; }}
        </style>

        <div id="gisapps-header">
            <h1>{title}</h1>
            <a href="{portal_url}">← All Apps</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
