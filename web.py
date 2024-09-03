import umap.umap_ as umap
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from eval import embed
import random
import hdbscan

with open('data/word') as f:
  # skip the first line
  examples = [line.strip() for line in f.readlines()]
  examples = random.choices(examples, k=1000)

distances = []

for b in examples:
    # Generate embeddings
    b_embed = embed(b)
    distances.append((b_embed, b))

# Extract the embeddings and labels
embeddings = [data[0] for data in distances]
labels = [data[1] for data in distances]

# Perform UMAP dimensionality reduction
reducer = umap.UMAP()

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Input(id='input-text', type='text', value='', placeholder='Enter text'),
    dcc.Graph(id='umap-plot', style={'width': '100%', 'height': 'calc(100vh - 50px)'})
], style={'height': '100vh'})

# Define callback to update plot based on input text
@app.callback(
    Output('umap-plot', 'figure'),
    [Input('input-text', 'value')]
)
def update_plot(input_text):
    global distances, embeddings, labels
    # Create Plotly figure
    fig = go.Figure()
    if input_text:
      # check if text is in distances
      exists = False
      for data in distances:
        if data[1] == input_text:
           exists = True

      if not exists:
        text_embed = embed(input_text)
        distances.append((text_embed, input_text))

        embeddings = [data[0] for data in distances]
        labels = [data[1] for data in distances]

    embedding_umap = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusters = clusterer.fit_predict(embedding_umap)

    # Add scatter plot with labels
    fig.add_trace(go.Scatter(
        x=embedding_umap[:, 0],
        y=embedding_umap[:, 1],
        mode='markers+text',
        marker=dict(color=clusters, colorscale='Viridis', opacity=0.5, size=8),
        text=labels + [input_text],  # Add input text to labels
        showlegend=False
    ))

    # highlight the input text
    if input_text:
        fig.add_trace(go.Scatter(
            x=[embedding_umap[-1, 0]],
            y=[embedding_umap[-1, 1]],
            mode='markers+text',
            marker=dict(color='red', size=10),
            text=[input_text],
            showlegend=False
        ))
        

    # Update layout
    fig.update_layout(
        title='UMAP Projection of Embeddings with Labels',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
