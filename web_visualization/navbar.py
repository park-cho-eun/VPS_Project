from dash import html
import dash_bootstrap_components as dbc


def create_navbar():
    navbar = dbc.Navbar([dbc.Container([
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [dbc.Col(dbc.NavbarBrand("Visual Public Sentiment Project")),],
                    align="center",
                    className="g-0",
                ),
                href="http://127.0.0.1:8050/",
                style={"textDecoration": "none"},
            ),
        dbc.NavItem(dbc.NavLink("전세사기", href="/")),
        dbc.NavItem(dbc.NavLink("스토킹처벌", href="/stalking")),
        dbc.NavItem(dbc.NavLink("청년정책", href="/youth")),
        dbc.NavItem(dbc.NavLink("간호법", href="/nurse"))
        ],
    )], color="dark", dark=True, fixed='top')

    return navbar