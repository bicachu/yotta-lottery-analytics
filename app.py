import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ClientsideFunction

import pandas as pd
import pathlib
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
from collections import Counter


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

# path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# read data
df = pd.read_csv(DATA_PATH.joinpath("yotta_numbers.csv"), parse_dates=['EndingDate'], infer_datetime_format=True)
last_updated = df.EndingDate.max().date()  # store last updated date

# filter and halve data set to recent weeks only
mid_date = df.EndingDate.median()  # store mid-point date from data set
filtered_df = df[df['EndingDate'] >= mid_date]

# get count of high momentum numbers and selected yotta numbers
count_high_mom_nums = \
    len(set(filtered_df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].values.T.ravel().tolist()))

count_selected_yottas = len(set(df[['YottaBallNumber']].values.T.ravel().tolist()))

# create ticket data frame shell
tickets_df = pd.DataFrame([{'Ticket': i + 1,
                            'Number 1': '-',
                            'Number 2': '-',
                            'Number 3': '-',
                            'Number 4': '-',
                            'Number 5': '-',
                            'Number 6': '-',
                            'Yotta Ball': '-'} for i in range(100)])


def chunks(lst, n):
    """
    :param: lst: list of values.
    :param: n: desired number of values in each chunk.

    :return: list of n lists(matrix).
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_numbers_lists(df):
    """
    :param: df: filtered dataframe of all winning numbers in selected timeframe

    :return: list of regular numbers to randomly select from based on high-low momentum strategy
    """
    selected_recent_numbers = df[
        ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6']].values.T.ravel().tolist()
    unselected_recent_numbers = list(set(list(range(1, 71))) - set(selected_recent_numbers))

    return selected_recent_numbers, unselected_recent_numbers


# create sets of high momentum and low momentum numbers
selected_recent_numbers, unselected_recent_numbers = get_numbers_lists(filtered_df)


def get_yotta_list(df, approach):
    """
    :param: df: original dataframe of all winning numbers
    :param: approach: user input selected approach for generating yotta numbers.

    :return: list of yotta numbers to randomly select from
    """
    if approach == 'RAN':
        yotta_numbers = list(range(1, 64))
    elif approach == 'FAV-NON':
        selected_yotta_numbers = df[['YottaBallNumber']].values.T.ravel().tolist()
        unselected_yotta_numbers = list(set(list(range(1, 64))) - set(selected_yotta_numbers))
        yotta_numbers = selected_yotta_numbers + (2 * unselected_yotta_numbers)  # double the chances of choosing these
    else:  # approach is 'ONLY-NON'
        selected_yotta_numbers = df[['YottaBallNumber']].values.T.ravel().tolist()
        unselected_yotta_numbers = list(set(list(range(1, 64))) - set(selected_yotta_numbers))
        yotta_numbers = unselected_yotta_numbers

    return yotta_numbers


def get_number_counts(numbers):
    """
    :param: numbers: list of all numbers from all generated tickets

    :return: list of yotta numbers to randomly select from
    """
    frequency = Counter(numbers)
    return frequency


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Yotta Lottery Winning Numbers"),
            html.Div(
                id="intro",
                children="Explore past weekly numbers and yotta balls picked. Hover over each number in the heatmap to "
                         "view its frequency and last time it was selected in the lottery.",
            ),
        ],
    )


def generate_hm_control_card():
    """

    :return: A Div containing controls for heatmap graphs.
    """
    return html.Div(
        id="hm-control-card",
        children=[
            html.P("Select Date Range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=df.EndingDate.min(),
                end_date=df.EndingDate.max(),
                min_date_allowed=df.EndingDate.min(),
                max_date_allowed=df.EndingDate.max(),
                initial_visible_month=df.EndingDate.max(),
            ),
            html.Br(),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
            html.Br(),
            html.Br(),
            html.Footer(
                "Note: The data used goes back to the weekly numbers picked on 9/27/2020 for consistency purposes. "
                "This was the first week that Yotta Savings changed the lottery format to 70 regular numbers and "
                "63 Yotta Ball numbers.")
        ],
    )


def generate_table_control_card():
    """

    :return: A Div containing controls for ticket selection table.
    """
    return html.Div(
        id="table-control-card",
        children=[
            html.H5("Generate Your Own Lucky Tickets"),
            html.Div(
                id='control-desc',
                children=[
                    "Select the number of tickets you are looking to generate numbers for as well as the "
                         "corresponding approach for both daily numbers and the yotta ball. The momentum approach is "
                         "based on the strategy described ",
                    html.A(
                        children=['here.'],
                        href='https://pub.towardsai.net/i-used-data-analytics-to-play-the-free-lottery-and-won-578e618e2711'),
                ]
            ),
            # html.Div(
            #     id="control-desc",
            #     children="Select the number of tickets you are looking to generate numbers for as well as the "
            #              "corresponding approach for both daily numbers and the yotta ball. The momentum approach is "
            #              "based on the strategy described here.",
            # ),
            html.P("Number of Tickets"),
            dcc.Input(
                id='n-tickets-input',
                inputMode='numeric',
                type='number',
                min=1,
                # max=10000,
                step=1,
                value=100,
            ),
            html.P("Number Selection Approach"),
            dcc.Dropdown(
                id='number-selection-approach',
                className='div-for-dropdown',
                clearable=False,
                options=[
                    {'label': 'Momentum',
                     'value': 'MOM',
                     'title': 'This approach selects a combo of high and low momentum numbers where high momentum '
                              'numbers are the ones that were picked at least once in the recent half of all '
                              'weekly drawings '
                     },
                    {'label': 'Random',
                     'value': 'RAN',
                     'title': 'This approach selects numbers completely randomly - '
                              'same approach the Yotta Savings app uses. '
                     }],
                value='MOM',
                placeholder='Select a strategy for picking daily first 6 numbers'
            ),
            html.Div(
                id='momentum-panel',
                children=[
                    html.Footer("*Currently, " + str(count_high_mom_nums) +
                                " high momentum numbers and " + str(70 - count_high_mom_nums) +
                                " low momentum numbers exist."),
                    html.Div(
                        id='momentum-inputs',
                        children=[
                            html.H6("High Momentum Numbers:"),
                            dcc.Dropdown(
                                id='high-momentum-input',
                                className='high-momentum-input',
                                clearable=False,
                                options=[{"label": i, "value": i} for i in range(0, 7)],
                                value=3),
                            html.H6("Low Momentum Numbers:"),
                            dcc.Input(
                                id='low-momentum-input',
                                inputMode='numeric',
                                type='number',
                                disabled=True,
                                min=0,
                                max=6,
                                step=1,
                                value=3,
                                style={'width': '12%'}
                            )]),
                ]),
            html.P("Yotta Ball Selection Approach"),
            dcc.Dropdown(
                id='yotta-selection-approach',
                className='div-for-dropdown',
                clearable=False,
                options=[
                    {'label': 'Favor Non-Selected Numbers',
                     'value': 'FAV-NON',
                     'title': 'This approach selects from all possible numbers but favors those that have never '
                              'before been selected'
                     },
                    {'label': 'Only Non-Selected Numbers',
                     'value': 'ONLY-NON',
                     'title': 'This approach only selects from numbers that have never been '
                              'picked as Yotta Ball before. '
                     },
                    {'label': 'Random',
                     'value': 'RAN',
                     'title': 'This approach selects Yotta ball number completely randomly - '
                              'same approach the Yotta Savings app uses. '
                     }],
                value='FAV-NON',
                placeholder='Select a strategy for picking the Yotta Ball'
            ),
            html.Footer("*Currently, " + str(63 - count_selected_yottas) +
                        " out of 63 Yotta Balls have never been selected."),
            html.Br(),
            html.Div(
                id="submit-btn-outer",
                children=html.Button(
                    id="submit-btn",
                    children="Generate",
                    n_clicks=0),
            ),
        ],
    )


def generate_heatmap(start, end):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    :param: reset (boolean): reset heatmap graph if True.

    :return: number count annotated heatmap.
    """

    filtered_df = df[(df.EndingDate >= start) & (df.EndingDate <= end)]
    reg_dates_dict = dict.fromkeys(range(1, 71), None)
    reg_counts_dict = dict.fromkeys(range(1, 71), 0)
    yotta_dates_dict = dict.fromkeys(range(1, 64), None)
    yotta_counts_dict = dict.fromkeys(range(1, 64), 0)

    for row in filtered_df.itertuples():
        for i in range(2, 8):
            reg_counts_dict[row[i]] += 1
            if reg_dates_dict[row[i]] is None:
                reg_dates_dict[row[i]] = row.EndingDate.date()
            else:
                reg_dates_dict[row[i]] = max(row.EndingDate.date(), reg_dates_dict[row[i]])
        yotta_counts_dict[row.YottaBallNumber] += 1
        if yotta_dates_dict[row.YottaBallNumber] is None:
            yotta_dates_dict[row.YottaBallNumber] = row.EndingDate.date()
        else:
            yotta_dates_dict[row.YottaBallNumber] = max(row.EndingDate.date(), yotta_dates_dict[row.YottaBallNumber])

    reg_counts = list(reg_counts_dict.values())
    reg_numbers = list(reg_counts_dict.keys())
    yotta_counts = list(yotta_counts_dict.values())
    yotta_numbers = list(yotta_counts_dict.keys())

    reg_counts_matrix = list(chunks(reg_counts, 10))
    yotta_counts_matrix = list(chunks(yotta_counts, 9))
    reg_numbers_matrix = list(chunks(reg_numbers, 10))
    yotta_numbers_matrix = list(chunks(yotta_numbers, 9))

    # create custom hover templates
    hovertext_reg = []
    for row in reg_numbers_matrix:
        hovertext_reg.append(list())
        for num in row:
            hovertext_reg[-1].append('<b># Times Picked:</b> {}<br />'
                                     '<b>Last Picked:</b> {}'.format(reg_counts_dict[num], reg_dates_dict[num]))

    hovertext_yotta = []
    for row in yotta_numbers_matrix:
        hovertext_yotta.append(list())
        for num in row:
            hovertext_yotta[-1].append('<b># Times Picked:</b> {}<br />'
                                       '<b>Last Picked:</b> {}'.format(yotta_counts_dict[num], yotta_dates_dict[num]))

    fig_nums = ff.create_annotated_heatmap(
        z=reg_counts_matrix[::-1],
        annotation_text=reg_numbers_matrix[::-1],
        font_colors=['black', 'white'],
        colorscale='purp')

    # add color scale
    fig_nums['data'][0]['showscale'] = True

    fig_nums.update_layout(
        coloraxis=dict(
            colorbar=dict(
                tickmode='array',
                ticktext=list(set(reg_counts)),
                tickvals=list(set(reg_counts)),
                nticks=3,
                thickness=15,
                xpad=3)),
        font=dict(
            size=15
        ),
        margin=dict(l=20, r=0, b=10, t=10, pad=2),
        xaxis=dict(fixedrange=True),
        xaxis_showgrid=False,
        yaxis=dict(fixedrange=True),
        yaxis_showgrid=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig_nums.update_traces(
        xgap=3,
        ygap=3,
        hovertemplate=None,
        hoverinfo='text',
        text=hovertext_reg[::-1]
    )

    fig_yotta = ff.create_annotated_heatmap(
        z=yotta_counts_matrix[::-1],
        annotation_text=yotta_numbers_matrix[::-1],
        font_colors=['black', 'white'],
        colorscale='reds')

    # add color scale
    fig_yotta['data'][0]['showscale'] = True

    fig_yotta.update_layout(
        coloraxis=dict(
            colorbar=dict(
                tickmode='array',
                ticktext=list(set(yotta_counts)),
                tickvals=list(set(yotta_counts)),
                nticks=3,
                thickness=15,
                xpad=3)),
        font=dict(
            size=15
        ),
        margin=dict(l=20, r=0, b=10, t=10, pad=2),
        xaxis=dict(fixedrange=True),
        xaxis_showgrid=False,
        yaxis=dict(fixedrange=True),
        yaxis_showgrid=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig_yotta.update_traces(
        xgap=3,
        ygap=3,
        hovertemplate=None,
        hoverinfo='text',
        text=hovertext_yotta[::-1]
    )

    return fig_nums, fig_yotta


def generate_right_col_content(n_clicks, selected_tab, n_tickets, number_selection, yotta_selection, high_mom):
    if selected_tab == 'lucky' and n_clicks == 0:
        return html.Div(
            children=[
                html.Br(),
                html.B('Choose configuration and generate data',
                       style={'color': 'red'}),
                html.Br(),
                html.Br(),
                dash_table.DataTable(
                    id='table',
                    data=tickets_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in tickets_df.columns],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(81, 81, 81)',
                                  'color': 'white'},
                    page_size=25,
                    fixed_rows={'headers': True},
                )])
    elif selected_tab == 'lucky' and n_clicks > 0:
        return generate_tickets_table(n_tickets, number_selection, yotta_selection, high_mom)

    else:
        return [html.Div(
            id="regular_numbers_card",
            children=[
                html.B("REGULAR NUMBERS",
                       style={'color': '#3916af'}),
                dcc.Graph(id="numbers_hm"),
            ],
        ),

            html.Div(
                id="yotta_ball_card",
                children=[
                    html.B("YOTTA BALLS",
                           style={'color': '#3916af'}),
                    dcc.Graph(id='yotta_hm'),
                ],
            )
        ]


def get_random_ticket():
    s = set()
    while len(s) < 6:
        s.add(random.choice(range(1, 71)))
    random_nums = list(s)
    random_nums.sort()

    return random_nums


def generate_tickets_table(n_tickets, number_selection, yotta_selection, high_mom):
    generated_tickets = []
    yotta_numbers = get_yotta_list(filtered_df, yotta_selection)

    for t in range(n_tickets):
        s = set()
        if number_selection == 'MOM':
            while len(s) < high_mom:
                s.add(random.choice(selected_recent_numbers))
            while len(s) < 6:
                s.add(random.choice(unselected_recent_numbers))
            random_nums = list(s)
            random_nums.sort()
        elif number_selection == 'RAN':
            random_nums = get_random_ticket()

        if yotta_selection == 'FAV-NON' or yotta_selection == 'ONLY-NON':
            YottaFound = False  # set this to True once a unique Yotta number is picked
            while (YottaFound == False):
                y_ball = random.choice(yotta_numbers)  # only select yotta ball numbers that have not yet been selected
                if y_ball not in s:
                    YottaFound = True

        elif yotta_selection == 'RAN':
            y_ball = random.choice(yotta_numbers)

        random_nums.append(y_ball)
        random_nums.insert(0, 'Ticket ' + str(t + 1))
        generated_tickets.append(random_nums)

    colnames = ['Ticket', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Number 6', 'Yotta Ball']
    tickets_df = pd.DataFrame(generated_tickets, columns=colnames)

    all_nums = tickets_df[
        ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Number 6']].values.T.ravel().tolist()
    all_yottas = tickets_df[['Yotta Ball']].values.T.ravel().tolist()
    num_counts = get_number_counts(all_nums)
    yotta_counts = get_number_counts(all_yottas)

    most_frequent_numbers = [i[0] for i in num_counts.most_common(3)]
    most_frequent_yotta = yotta_counts.most_common(1)[0][0]
    most_frequent_numbers.append(most_frequent_yotta)

    return html.Div(
        children=[
            html.H5('Your Lucky Tickets'),
            dash_table.DataTable(
                id='table',
                data=tickets_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in tickets_df.columns],
                style_header={'backgroundColor': 'rgb(81, 81, 81)',
                              'color': 'white'},
                style_cell={'textAlign': 'center'},
                page_size=25,
                fixed_rows={'headers': True},
                export_format='csv'
            ),
            html.Br(),
            html.H5('Numbers To Root For'),
            dcc.Graph(
                id='top-number-circles',
                figure=generate_top_numbers_bubbles(most_frequent_numbers)),
        ])


def generate_top_numbers_bubbles(top_numbers):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0.25, 0.85, 1.45, 2.05],
        y=[0.25, 0.25, 0.25, 0.25],
        text=top_numbers,
        mode="text",
        textfont=dict(
            color="black",
            size=26,
        ),
        hoverinfo='text'
    ))

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    # add circles
    fig.add_shape(type="circle",
                  line_color="black",
                  fillcolor="yellow",
                  x0=0,
                  y0=0,
                  x1=0.5,
                  y1=0.50,
                  opacity=0.5
                  )
    fig.add_shape(type="circle",
                  line_color="black",
                  fillcolor="yellow",
                  x0=0.60,
                  y0=0,
                  x1=1.1,
                  y1=0.50,
                  opacity=0.5
                  )

    fig.add_shape(type="circle",
                  line_color="black",
                  fillcolor="yellow",
                  x0=1.2,
                  y0=0,
                  x1=1.7,
                  y1=0.50,
                  opacity=0.5
                  )
    fig.add_shape(type="circle",
                  line_color="black",
                  fillcolor="red",
                  x0=1.8,
                  y0=0,
                  x1=2.3,
                  y1=0.50,
                  opacity=0.5
                  )

    fig.update_layout(
        margin=dict(l=0, r=0, b=150, t=50),
        xaxis=dict(fixedrange=True),
        xaxis_showgrid=False,
        yaxis=dict(fixedrange=True),
        yaxis_showgrid=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',

    )
    return fig


app.layout = html.Div(
    id="app-container",
    children=[
        # banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Div(
                    style={'display': 'flex',
                           'flex-direction': 'row',
                           'align-items': 'center'},
                    children=[
                        html.H2("Yotta Savings Lottery Dashboard"),
                        html.H6("(Last updated: " + str(last_updated) + ")")]),
                html.A(
                    children=[
                        html.Img(
                            src='https://img.buymeacoffee.com/button-api/?text=Support this project&emoji=&slug='
                                'datacaffeine&button_colour=BD5FFF&font_colour=ffffff&font_family=Lato&outline_colour'
                                '=000000&coffee_colour=FFDD00',
                        )],
                    href='https://www.buymeacoffee.com/datacaffeine')
            ]
        ),
        # left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[dcc.Tabs(
                id='tabs-bar',
                className='tabs-bar',
                value='overview',  # default tab
                children=[
                    dcc.Tab(
                        label='About',
                        className='control-tab',
                        selected_className='control-tab--selected',
                        value='overview',
                        children=[
                            html.H5('Overview'),
                            html.P('This application tracks the historical numbers picked for the weekly  '
                                   'Yotta Savings lottery. '),
                            html.Li(
                                'To explore the most frequent numbers and yotta balls selected, view the '
                                'Dashboard page.   '),
                            html.Li(
                                'To generate your own random weekly "lucky" numbers  based on '
                                'data analysis of the patterns observed with high and low momentum numbers, '
                                'view the Pick Lucky # page.'),
# 'These numbers are picked based on the strategies described here.
                            html.Br(),
                            html.Div(
                                children=[
                                    'For more information about Yotta Savings, view their website ',
                                    html.A(
                                        children=['here.'],
                                        href='https://www.withyotta.com/'),
                                    html.Br(),
                                    ' And ',
                                    html.A(
                                        children=['sign up'],
                                        href='https://members.withyotta.com/register?code=DATA'),
                                    ' using the invite code ',
                                    html.Strong('DATA'),
                                    ' and you will get 100 free tickets for your first week!'
                                ]
                            ),
                            html.Br(),
                            html.Div(
                                children=[
                            html.H6('Developed with: '),
                            html.A(
                                children=[
                                    html.Img(
                                        src='assets/dash-logo.png',
                                        style={'width': '20%'}
                                    )],
                                href='https://plotly.com/dash/')])
                        ]),
                    dcc.Tab(
                        label='Dashboard',
                        className='control-tab',
                        selected_className='control-tab--selected',
                        value='dashboard',
                        children=[description_card(), generate_hm_control_card()]
                                 + [html.Div(
                            ["initial child"], id="output-clientside", style={"display": "none"}
                        )
                                 ],
                    ),
                    dcc.Tab(
                        label='Pick Lucky #',
                        className='lucky-tab',
                        selected_className='control-tab--selected',
                        value='lucky',
                        children=[generate_table_control_card()]
                    ),
                ]
            )]

        ),
        # right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                dcc.Loading(
                    children=[
                        html.H3('Please generate data'),
                        html.Br(),
                        dash_table.DataTable(
                            id='tickets-table',
                            data=tickets_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in tickets_df.columns],
                            style_cell={'textAlign': 'center'},
                            page_size=25,
                            fixed_rows={'headers': True},
                        )],
                    id="tickets-table-content",
                    color="rgb(103, 26, 248)",
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("tickets-table-content", "children"),
    [Input("tabs-bar", "value"),
     Input("submit-btn", "n_clicks")],
    state=[State("n-tickets-input", "value"),
           State("number-selection-approach", "value"),
           State("yotta-selection-approach", "value"),
           State("high-momentum-input", "value"),
           ]
)
def update_right_col_content(selected_tab, n_clicks, n_tickets, number_selection, yotta_selection, high_mom):
    return generate_right_col_content(n_clicks, selected_tab, n_tickets,
                                      number_selection, yotta_selection, high_mom)


@app.callback(
    [Output("date-picker-select", "start_date"),
     Output("date-picker-select", "end_date")],
    Input("reset-btn", "n_clicks"),
)
def reset_dates(reset_click):
    if reset_click:
        return df.EndingDate.min(), df.EndingDate.max()
    else:
        return df.EndingDate.min(), df.EndingDate.max()


@app.callback(
    [Output("momentum-panel", "style"),
     Output("momentum-inputs", "style")],
    Input("number-selection-approach", "value")
)
def show_hide_momentum_inputs(approach_selected):
    if approach_selected == 'MOM':
        return {'display': 'block'}, {'display': 'inline-flex'}
    if approach_selected == 'RAN':
        return {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output("low-momentum-input", "value"),
    Input("high-momentum-input", "value")
)
def update_low_momentum_value(high_momentum_tickets):
    return 6 - high_momentum_tickets


@app.callback(
    [Output("numbers_hm", "figure"),
     Output("yotta_hm", "figure")],
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
    ],
)
def update_heatmap(start, end):
    return generate_heatmap(
        start, end
    )


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("numbers_hm", "figure"),
     Input("yotta_hm", "figure")]
)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
