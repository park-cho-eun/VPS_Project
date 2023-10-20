'''
 # @ Create Time: 2023-10-15 21:47:11.312221
'''

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from navbar import create_navbar

# Toggle the themes at [dbc.themes.LUX]
# The full list of available themes is:
# BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN,
# LUX, MATERIA, MINTY, PULSE, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR,
# SPACELAB, SUPERHERO, UNITED, YETI, ZEPHYR.
# To see all themes in action visit:
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/

NAVBAR = create_navbar()
# To use Font Awesome Icons

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX,],
    use_pages=True,  # New in Dash 2.7 - Allows us to register pages
)

app.layout = dcc.Loading(  # <- Wrap App with Loading Component
    id='loading_page_content',
    children=[
        html.Div(
            [
                NAVBAR,
                dash.page_container
            ]
        )
    ],
    color='primary',  # <- Color of the loading spinner
    fullscreen=True  # <- Loading Spinner should take up full screen
)

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
