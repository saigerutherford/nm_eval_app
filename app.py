import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import pandas as pd
import base64
import io

data_path = 'https://raw.githubusercontent.com/saigerutherford/brainviz-app/main/cross_validation_10fold_evaluation.csv'

# Load Data
def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(by="Label")
    return df

df = load_data(data_path)

# Colors & Styling
colors = {
    'background': '#f4f4f4',
    'card': '#ffffff',
    'text': '#333333',
    'primary': '#007bff',
}

# Function to create a styled figure
def create_figure(df, y_col, y_label):
    fig = go.Figure(data=[
        go.Scatter(
            x=df["Label"],
            y=df[y_col],
            mode="markers",
            marker=dict(
                colorscale='plasma',
                color=df[y_col],
                size=12,
                line=dict(width=0.7, color='Black'),
                colorbar={"title": " "},
                reversescale=True,
                opacity=0.8,
            ),
            customdata=df[[y_col, "IMG_URL"]],
            hovertext=df["Label"],
            hoverinfo="text"
        )
    ])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_layout(
        autosize=True,
        height=300,
        width=450,
        xaxis=dict(title='ROI', showticklabels=False),
        yaxis=dict(title=y_label, showgrid=True),
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

# Create Figures
fig1 = create_figure(df, "EV", "Explained Variance (EV)")
fig2 = create_figure(df, "MSLL", "Mean Squared Log Loss")
fig3 = create_figure(df, "Skew", "Skew")
fig4 = create_figure(df, "Kurtosis", "Kurtosis")

app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '30px'}, children=[
    html.Div(html.H1("Normative Modeling Evaluation Metrics Viz",
                     style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '28px', 'fontWeight': 'bold',
                            'padding': '15px', 'backgroundColor': colors['card'], 'borderRadius': '10px',
                            'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'})),

    html.Div([
        html.Div([
            html.P("Upload a CSV file to visualize different normative modeling evaluation metrics.",
                   style={'fontSize': '16px', 'color': colors['text'], 'paddingBottom': '10px'}),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload CSV', style={'backgroundColor': colors['primary'],
                                                          'color': 'white',
                                                          'border': 'none',
                                                          'padding': '10px 15px',
                                                          'fontSize': '16px',
                                                          'cursor': 'pointer',
                                                          'borderRadius': '5px'}),
                multiple=False
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px',
                  'backgroundColor': colors['card'], 'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'}),

        html.Div([
            html.Div([
                html.Div([
                    html.H3("Explained Variance", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(id="graph1", figure=fig1, clear_on_unhover=True)
                ], style={'width': '35%', 'display': 'inline-block', 'padding': '20px'}),
                html.Div([
                    html.H3("Mean Squared Log Loss", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(id="graph2", figure=fig2, clear_on_unhover=True)
                ], style={'width': '35%', 'display': 'inline-block', 'padding': '20px'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            html.Div([
                html.Div([
                    html.H3("Skew", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(id="graph3", figure=fig3, clear_on_unhover=True)
                ], style={'width': '35%', 'display': 'inline-block', 'padding': '20px'}),
                html.Div([
                    html.H3("Kurtosis", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(id="graph4", figure=fig4, clear_on_unhover=True)
                ], style={'width': '35%', 'display': 'inline-block', 'padding': '20px'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dcc.Tooltip(id="graph-tooltip"),
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'})
])

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    [Input("graph1", "hoverData"),
     Input("graph2", "hoverData"),
     Input("graph3", "hoverData"),
     Input("graph4", "hoverData")],
    prevent_initial_call=True
)
def display_hover(hoverData1, hoverData2, hoverData3, hoverData4):
    hover_data = next((hover for hover in [hoverData1, hoverData2, hoverData3, hoverData4] if hover is not None), None)
    if hover_data is None:
        return False, no_update, no_update
    pt = hover_data["points"][0]
    bbox = pt["bbox"]
    metric_value, img_src = pt["customdata"]
    name = pt["x"]
    children = html.Div([
        html.Img(src=img_src, style={"width": "80%"}),
        html.H2(f"{name}", style={"color": "darkblue"}),
        html.P(f"Value = {metric_value:.3f}"),
    ], style={'width': '250px', 'white-space': 'normal'})
    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=False)