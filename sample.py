import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

quarters = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
bubble_data = {
    "부동산 문제": [10, 20, 30, 60, 90, 120],
    "깡동전세 문제": [8, 30, 40, 50, 120, 150],
    "역전세 문제": [5, 8, 25, 60, 100, 110],
    "피해 현황": [12, 50, 180, 100, 80, 50],
    "수사 현황": [6, 10, 30, 90, 120, 80],
    "정치권의 대응": [9, 11, 40, 150, 200, 100],
    "원인 지적": [7, 10, 12, 60, 70, 90],
    "대응 지적": [8, 7, 50, 100, 110, 120]
}

df = pd.DataFrame(bubble_data, index=quarters)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='bubble-chart'),
    dcc.Slider(
        id='quarter-slider',
        min=0,
        max=len(quarters) - 1,
        step=1,
        value=0,
        marks={i: quarter for i, quarter in enumerate(quarters)}
    ),
    html.Div([
        html.Div(id='pie-div', children=[], style={'width': '50%', 'display': 'inline-block'}),
        html.Div(id='content-div', children=[], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})  # 수정된 스타일
    ]),
    html.Button('지우기', id='clear-button', n_clicks=0)
])

@app.callback(
    Output('bubble-chart', 'figure'),
    [Input('quarter-slider', 'value'),
     Input('clear-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_bubble_chart(selected_quarter, clear_clicks):
    selected_quarter_data = df.iloc[selected_quarter]
    
    bubble_trace = []
    x_positions = [1, 3, 5,   8, 10, 12,   15, 17]  # 수정된 X 좌표
    
    for idx, column in enumerate(df.columns):
        x_position = x_positions[idx]
        y_position = selected_quarter_data[column]
        
        bubble_trace.append(go.Scatter(
            x=[x_position],
            y=[y_position],
            mode='markers',
            marker=dict(size=y_position),
            name=column,
            text=column,
            hoverinfo='text+x+y'
        ))

    layout = go.Layout(
        title={'text' : '전세사기 이슈',
               'x': 0.01,  # 제목 위치 조정
               'xanchor': 'left',  # 제목 위치 설정
               'font': {'size': 35},  # 원하는 글자 크기로 변경
               'y': 0.95  # 제목 위치 조정
        },
        xaxis=dict(title='', showticklabels=False),
        yaxis=dict(title='', showticklabels=False),
        hovermode='closest',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        legend=dict(orientation='v'),
    )

    annotations = []
    if clear_clicks % 2 == 1:
        layout['annotations'] = []
    else:
        for idx, column in enumerate(df.columns):
            annotations.append(
                dict(
                    x=x_positions[idx],
                    y=selected_quarter_data[column],
                    text=column,
                    xref="x",
                    yref="y",
                    showarrow=False,
                    font=dict(size=14)
                )
            )
        layout['annotations'] = annotations

    fig = go.Figure(data=bubble_trace, layout=layout)
    return fig

@app.callback(
    [Output('pie-div', 'children'),
     Output('content-div', 'children')],
    [Input('bubble-chart', 'clickData'),
     Input('quarter-slider', 'value')],
    prevent_initial_call=True
)
def create_pie_chart(click_data, selected_quarter):
    pie_children = []
    content_children = []
    
    if click_data and 'curveNumber' in click_data['points'][0] and 'pointNumber' in click_data['points'][0]:
        curve_number = click_data['points'][0]['curveNumber']
        point_number = click_data['points'][0]['pointNumber']
        label = df.columns[curve_number]
        
        if label == '대응 지적':
            selected_quarter_data = df.iloc[point_number]
            
            total = selected_quarter_data.sum()
            
            # Calculate custom ratios based on the selected quarter
            if selected_quarter == 0 :
                ratio_a = 10  # 임의의 비율 조정
                ratio_b = 10
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 1 :
                ratio_a = 15  # 임의의 비율 조정
                ratio_b = 20
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 2 :
                ratio_a = 25  # 임의의 비율 조정
                ratio_b = 30
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 3 :
                ratio_a = 45  # 임의의 비율 조정
                ratio_b = 30
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 4 :
                ratio_a = 45  # 임의의 비율 조정
                ratio_b = 50
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 5 :
                ratio_a = 40  # 임의의 비율 조정
                ratio_b = 40
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '민주당이 추진하는 특별법은 보증금 미반환 주택의 범위가 너무 넓게 설정되어 있고, 국가가 전세 채권을 공공 매입하는 기준이 명확하지 않다는 비판을 받고 있습니다. 이로 인해 해당 법안의 현실적 실행 가능성에 대한 의문이 제기되고 있습니다.'
                text_b = '피해자들은 이러한 지원책이 실질적인 도움보다는 겉으로만 보여주기 위한 것이라며 비판하였습니다. 그들은 전세사기 피해자로 인정받는 기준을 더 완화할 것을 요구하고 있습니다.'
                text_c = '각 정당이 자신들의 기존 입장을 고수하며 상호간의 의견 차이를 좁히지 못하고 있습니다. 이로 인해 전세사기 특별법은 국토교통위원회에서 일주일 넘게 진행되지 못하고 있는 상황입니다. 한편, 이런 지연으로 인해 전세사기 피해자들의 고통은 계속해서 증가하고 있습니다.'
            
            pie_trace = go.Pie(labels=['민주당', '국민의힘', '이외'], values=ratios)
            pie_layout = go.Layout(
                title=label,
                plot_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            pie_fig = go.Figure(data=[pie_trace], layout=pie_layout)
            pie_children.append(dcc.Graph(id='hidden-pie-chart', figure=pie_fig))
            
            content_children.append(
                html.Div([
                    html.Div([
                        html.H2('민주당'),
                        html.P(text_a)
                    ], style={'margin-bottom': '20px'}),  # 제목과 본문 사이 간격 설정

                    html.Div([
                        html.H2('국민의힘'),
                        html.P(text_b)
                    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),  # 각 그룹 간 간격 설정

                    html.Div([
                        html.H2('이외'),
                        html.P(text_c)
                    ], style={'margin-top': '20px'})  # 제목과 그룹 사이 간격 설정
                ])
            )
            
        elif label == '원인 지적':
            selected_quarter_data = df.iloc[point_number]
            
            total = selected_quarter_data.sum()
            
            # Calculate custom ratios based on the selected quarter
            if selected_quarter == 0 :
                ratio_a = 10  # 임의의 비율 조정
                ratio_b = 10
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 1 :
                ratio_a = 15  # 임의의 비율 조정
                ratio_b = 20
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 2 :
                ratio_a = 25  # 임의의 비율 조정
                ratio_b = 30
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 3 :
                ratio_a = 45  # 임의의 비율 조정
                ratio_b = 30
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 4 :
                ratio_a = 45  # 임의의 비율 조정
                ratio_b = 50
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '의견1'
                text_b = '의견2'
                text_c = '의견3'
                
            elif selected_quarter == 5 :
                ratio_a = 70  # 임의의 비율 조정
                ratio_b = 20
                ratio_c = 100 - ratio_a - ratio_b
                ratios = [ratio_a, ratio_b, ratio_c]
                text_a = '민주당이 추진하는 특별법은 보증금 미반환 주택의 범위가 너무 넓게 설정되어 있고, 국가가 전세 채권을 공공 매입하는 기준이 명확하지 않다는 비판을 받고 있습니다. 이로 인해 해당 법안의 현실적 실행 가능성에 대한 의문이 제기되고 있습니다.'
                text_b = '피해자들은 이러한 지원책이 실질적인 도움보다는 겉으로만 보여주기 위한 것이라며 비판하였습니다. 그들은 전세사기 피해자로 인정받는 기준을 더 완화할 것을 요구하고 있습니다.'
                text_c = '각 정당이 자신들의 기존 입장을 고수하며 상호간의 의견 차이를 좁히지 못하고 있습니다. 이로 인해 전세사기 특별법은 국토교통위원회에서 일주일 넘게 진행되지 못하고 있는 상황입니다. 한편, 이런 지연으로 인해 전세사기 피해자들의 고통은 계속해서 증가하고 있습니다.'
            
            pie_trace = go.Pie(labels=['임대차 3법', 'A법', 'B법'], values=ratios)
            pie_layout = go.Layout(
                title=label,
                plot_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            pie_fig = go.Figure(data=[pie_trace], layout=pie_layout)
            pie_children.append(dcc.Graph(id='hidden-pie-chart', figure=pie_fig))
            
            content_children.append(
                html.Div([
                    html.Div([
                        html.H2('임대차 3법'),
                        html.P(text_a)
                    ], style={'margin-bottom': '20px'}),  # 제목과 본문 사이 간격 설정

                    html.Div([
                        html.H2('A법'),
                        html.P(text_b)
                    ], style={'margin-top': '20px', 'margin-bottom': '20px'}),  # 각 그룹 간 간격 설정

                    html.Div([
                        html.H2('B법'),
                        html.P(text_c)
                    ], style={'margin-top': '20px'})  # 제목과 그룹 사이 간격 설정
                ])
            )
    
    return pie_children, content_children

if __name__ == '__main__':
    app.run_server(debug=True)

