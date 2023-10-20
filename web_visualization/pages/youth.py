import pandas as pd
import dash
from dash import dcc
from dash import html, register_page
import dash_bootstrap_components as dbc
from dash import Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px


#청년정책 데이터 불러오기
youth_n = pd.read_excel('assets/청년정책_연도별_대분류_기사개수.xlsx')
youth_n_det = pd.read_excel('assets/청년정책_연도별_소분류_기사개수.xlsx')
youth_cores = pd.read_json('assets/청년정책_연도별_대분류별_핵심기사요약.json')
youth_n.columns = ['num', 'year', 'big_cls', 'count']
youth_n_det.columns = ['num', 'year', 'small_cls', 'count']

youth_n['name'] = ['대응', '선거', '평가', '법률 및 정책']*5
youth_n_det['name'] = ['청년 기구 설치', '청년 인사 선발', '소통 및 행사', '예산 편성', '기타',
                       '컬럼', '통계', '인터뷰', '일자리', '주거', '교육', '복지', '기타']*5

youth_n['x'] = [-2, -1, -4, 5, -2, -1, -4, 5, -2, -1, -4, 5, -2, -1, -4, 5, -2, -1, -4, 5]
youth_n['y'] = [3, -4, 6, -1, 3, -4, 6, -1, 3, -4, 6, -1, 3, -4, 6, -1, 3, -4, 6, -1]
youth_n_det['num'] = [0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4, 6, 6, 6, 7, 7, 7, 7, 7,
                      8, 8, 8, 8, 8, 10, 10, 10, 11, 11, 11, 11, 11,
                      12, 12, 12, 12, 12, 14, 14, 14, 15, 15, 15, 15, 15,
                      16, 16, 16, 16, 16, 18, 18, 18, 19, 19, 19, 19, 19]

#웹실행
cards = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Custom CSS", className="card-title"),
            html.P(
                "This card has inline styles applied controlling the width. "
                "You could also apply the same styles with a custom CSS class."
            ),
        ]
    ),

)

layout_youth = html.Div([
dbc.Container([html.Br(), html.Br(), html.Br(), html.Br()],fluid=True),

    dbc.Container([
        html.A([dbc.Container([dbc.Row(
        [dbc.Col(html.Img(id='youth-temp', style={"height":"450px", "padding-right":20, "margim-right":30}), md=2),

            dbc.Col([dcc.Graph(id='youth-clust'),
         html.Div([dcc.Slider(19, 23, value=19, step=1,
                              marks={19:{'label':'19'}, 20:{'label':'20'}, 21:{'label':'21'}, 22:{'label':'22'}, 23:{'label':'23'}},
                              id = 'youth-slider')
                  ],style={"marginTop":"20px"})],
                 style={"padding-left":10, "padding-right":10, "padding-bottom":10, "padding-top":2, "marginTop":"5px"}
                 , md=4),
         dbc.Col([
             dbc.Row([dcc.Graph(id='youth-hist')], style={'height':'50%'}),
            dbc.Row(id='youth-news', style={'height':'40%', "width":"95%"}),
            ],
                 style={"padding-left":10, "padding-right":10, "padding-bottom":10, "padding-top":2, "marginTop":"5px"},
                 md=6),
         ], style={"padding":10})],
    )], id='youth',)
    ],fluid=True),

],)

@callback(Output('youth-clust', 'figure'),Input('youth-slider', 'value'))
def youth_update_clust(selected_qt):
    year = selected_qt + 2000
    data = youth_n[youth_n['year']==year]
    fig = go.Figure(go.Scatter(x=data['x'], y=data['y'], mode='markers+text', customdata=data['num'],
                               text=data['name'], textposition="middle center", textfont=dict(size=23),
                               marker=dict(size=data['count'], color=['#2772db', '#2eb872', '#facf5a', '#ff5959']),
                               ))

    fig.update_layout(go.Layout(paper_bgcolor='#ededed', plot_bgcolor='#ededed'))

    fig.update_xaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
    fig.update_yaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
    fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(showlegend=False)
    fig.update_layout(clickmode='event+select')
    return fig

@callback(Output('youth-news', 'children'),Input('youth-slider', 'value'),Input('youth-clust', 'clickData'))
def youth_update_cards(selected_qt, clickData):
    if clickData is None:
        cards = dbc.Card([
            dbc.CardBody(
                [html.H4("관련 대표 기사", className="card-title"),
                 html.Br(),
                 html.P('좌측 그림의 군집을 클릭해주세요.'),],
            ),
        ], style={"margin":30})

        return cards

    else:
        year = selected_qt + 2000
        n = clickData['points'][0]['customdata']
        data = youth_cores[youth_cores['year']==year]

        n = n+1
        if n%4==0:
            i = 40
            data = data[data['big_cls'] == i]['summs'].tolist()
            txt = list(data[0].values())
            txt = ' '.join(txt)
            txt = txt[:150]+str('···')
            title = '평가 관련 대표 기사'
        elif n%4==1:
            i = 10
            data = data[data['big_cls'] == i]['summs'].tolist()
            txt = list(data[0].values())
            txt = ' '.join(txt)
            txt = txt[:150]+str('···')
            title = '대응 관련 대표 기사'
        elif n%4==3:
            i = 30
            data = data[data['big_cls'] == i]['summs'].tolist()
            txt = list(data[0].values())
            txt = ' '.join(txt)
            txt = txt[:150]+str('···')
            title = '평가 관련 대표 기사'
        else:
            txt = '선거와 관련한 대표기사는 제공되지 않습니다.'
            title = '선거 관련 대표 기사'

        cards = dbc.Card([
            dbc.CardBody(
                [html.H4(title, className="card-title"),
                 html.P(txt),],
            ),
        ], style={"margin":30})
        return cards

@callback(Output('youth-hist', 'figure'),Input('youth-clust', 'clickData'))
def youth_update_hist(clickData):
    if clickData is None:
        data = youth_n_det[youth_n_det['num'] == 0]
        fig = px.bar(data, x='count', y='name', color='name', orientation='h', text='name')
        fig.update_layout(go.Layout(plot_bgcolor='#ededed'))
        fig.update_layout(showlegend=False)
        fig.update_coloraxes(showscale=False)
        fig.update_yaxes(title=None, showgrid=True, visible=False)
        fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
        fig.update_xaxes(showgrid=True)
        return fig
    else:
        n = clickData['points'][0]['customdata']
        data = youth_n_det[youth_n_det['num']==n]

        fig = px.bar(data, x='count', y='name', color='name', orientation='h', text='name')
        fig.update_layout(go.Layout(plot_bgcolor='#ededed'))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None, showgrid=True, visible=False)
        fig.update_xaxes(showgrid=True)
        fig.update_coloraxes(showscale=False)
        fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
        return fig

@callback(Output('youth-temp', 'src'), Input('youth-slider', 'value'))
def home_upadate_temp(selected_qt):
    selected_qt = selected_qt - 18
    src = 'assets/youth_'+str(selected_qt)+'.png'
    return src

register_page(
    __name__,
    name='Youth',
    top_nav=True,
    path='/youth',
    layout=layout_youth
)