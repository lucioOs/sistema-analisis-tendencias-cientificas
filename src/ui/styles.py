import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 2.0rem;
            max-width: 1180px;
        }

        :root {
          --bg-soft: #f7f9fc;
          --card-bg: rgba(255,255,255,0.92);
          --card-border: rgba(15,23,42,0.10);
          --text-main: #0f172a;
          --text-muted: #475569;
          --brand: #2563eb;
          --brand-soft: rgba(37, 99, 235, 0.10);
          --ok-soft: rgba(22, 163, 74, 0.12);
          --warn-soft: rgba(245, 158, 11, 0.12);
        }

        [data-theme="dark"] {
          --bg-soft: #0f172a;
          --card-bg: rgba(30,41,59,0.70);
          --card-border: rgba(148,163,184,0.25);
          --text-main: #e2e8f0;
          --text-muted: #94a3b8;
          --brand: #60a5fa;
          --brand-soft: rgba(96,165,250,0.18);
          --ok-soft: rgba(22, 163, 74, 0.20);
          --warn-soft: rgba(245, 158, 11, 0.20);
        }

        .app-hero {
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 14px 16px;
            background: linear-gradient(180deg, var(--brand-soft), transparent), var(--card-bg);
            margin-bottom: 0.85rem;
        }

        .card {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 14px;
            background: var(--card-bg);
            box-shadow: 0 3px 10px rgba(2, 6, 23, 0.05);
        }

        .muted { color: var(--text-muted); font-size: 0.93rem; }

        .bigbtn button {
            width: 100%;
            height: 56px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 14px;
            border: 1px solid var(--card-border);
        }

        .mode-card {
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            border-radius: 12px;
            padding: 12px;
            min-height: 128px;
        }

        .mode-title {
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.15rem;
        }

        .mode-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 0.4rem;
            background: var(--brand-soft);
            color: var(--brand);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--card-border);
            padding: 10px;
            border-radius: 12px;
            background: var(--card-bg);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
