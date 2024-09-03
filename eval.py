import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import umap.umap_ as umap
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import datasets
import Levenshtein
import faiss

# from torchviz import make_dot
# Load the saved model
model = torch.load('./model/808/cnn/word/nt1000_nq1000/model.torch')
model.eval()
print(model)


# make_dot(model).render("attached", format="png")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def encode_string(input_data):
    try:
        (alphabet_size, max_length, char_ids, alphabet) = datasets.word2sig([input_data], 30)
        char_ids = char_ids[0]
        encode = np.zeros((alphabet_size, max_length), dtype=np.float32)
        encode[np.array(char_ids), np.arange(len(char_ids))] = 1
        return encode
    except Exception as e:
        print(e)
        print(input_data)


@torch.no_grad()
def embed(input_data):
    alphabet_size = 26
    max_length = 30
    data = np.zeros((1024, alphabet_size, max_length), dtype=np.float32)
    data[0] = encode_string(input_data)
    data = torch.from_numpy(data).to("cuda")
    output = model.embedding_net(data)
    return output.cpu().data.numpy()[0]

if __name__ == "__main__":
  # rad the data/word file and use line for line
  with open('data/word') as f:
      # skip the first line
      examples = [line.strip() for line in f.readlines()][:10000]


  distances = []
  for a in ['julian']:
      t1 = time.time()
      a_embed = embed(a)
      t2 = time.time()
      for b in examples:

          # Generate embeddings
          b_embed = embed(b)
          t3 = time.time()

          tt1 = time.time()
          # Compute the L2 distance between the two embeddings
          l2_distance = np.linalg.norm(a_embed - b_embed)
          tt2 = time.time()

          # compute the cosine similarity
          tt3 = time.time()
          dot = np.dot(a_embed, b_embed.T)
          norm_a = np.linalg.norm(a_embed)
          norm_b = np.linalg.norm(b_embed)
          cos_dist = min(1 - dot / (norm_a * norm_b), 1)
          tt4 = time.time()

          t4 = time.time()
          levenshtein_dist = Levenshtein.distance(a, b) / max(len(a), len(b))
          t5 = time.time()
          import math

          distances.append((a, b, abs(cos_dist - levenshtein_dist)
                          if not math.isnan(levenshtein_dist) else 1, a_embed, b_embed))

          def highlight_diff(sample, correct):
              if abs(sample - correct) < 0.1:
                  return f"{bcolors.OKGREEN}`{sample:.2f}` ({abs((sample) - correct):.2f}){bcolors.ENDC}"
              elif abs(sample - correct) > 0.1 and abs(sample - correct) < 0.20:
                  return f"{bcolors.OKCYAN}`{sample:.2f}` ({abs((sample) - correct):.2f}){bcolors.ENDC}"
              else:
                  return f"{bcolors.FAIL}`{sample:.2f}` ({abs((sample) - correct):.2f}){bcolors.ENDC}"

          report = f"""
    ---------------------------------------
    String 1        : `{a}` (`{a_embed[:5]}...` size={len(a_embed)}) (took {t2-t1:.2f} sec/embed @ {1 / (t2 - t1):.2f} embeds/sec batched: { 1024 / (t2 - t1):.2f} embeds/sec)
    String 2        : `{b}` (`{b_embed[:5]}...` size={len(b_embed)}) (took {t3-t2:.2f} sec/embed @ {1 / (t3 - t2):.2f} embeds/sec batched: { 1024 / (t2 - t1):.2f} embeds/sec)
    --------------------------------------- 
    l2 distance     : {highlight_diff(l2_distance, levenshtein_dist)}` ({abs(l2_distance - levenshtein_dist):.2f}) ({abs(tt2 - tt1):.2f} @ {1 / (tt2 - tt1 + 1e-10):.2f} sec/computation)
    Cosine distance : {highlight_diff(cos_dist, levenshtein_dist)} ({abs(tt4 - tt3):.2f} @ {1 / (tt4 - tt3 + 1e-10):.2f} sec/computation)
    Fact distance   : `{levenshtein_dist:.2f}` (inverted) ({1 / (t5-t4):.2f} s/computation)

    """
          # print(report)

  # print("Mean distance", np.mean(distances))
  # print("Median distance", np.median(distances))
  # print("Worst 1%", np.percentile(distances, 99))
  # print("Worst 0.1%", np.percentile(distances, 99.9))
  # print("Worst difference", np.max(distances))


  # Placeholder for your distances data. Replace it with your actual data.
  data = distances

  # Extract numerical distances from the tuples and ensure they are finite.
  # This step is crucial to remove any potential NaN or infinity values that could disrupt statistical computations.
  data = [float(item[2]) for item in data if np.isfinite(float(item[2]))]

  # Ensure there's data to process to avoid errors in an empty list scenario.
  if data:
      # Fit a normal distribution to the distances data.
      # Here, we find the mean (mu) and standard deviation (std) that best approximate the distances distribution.
      mu, std = norm.fit(data)

      # Plotting the distribution of distances as a histogram.
      # The 'density=True' parameter normalizes the histogram, making the area under the histogram sum to 1, similar to a probability density function.
      # density=false will print the absolute values and not the probability distribution.
      plt.hist(data, bins=30, alpha=0.6, color='g', density=True)

      # Preparing to overlay a normal distribution curve based on the fitted mean and standard deviation.
      xmin, xmax = plt.xlim()
      x = np.linspace(xmin, xmax, 100)
      p = norm.pdf(x, mu, std)
      plt.plot(x, p, 'k', linewidth=2)

      # Adding a title to the plot including the mean and standard deviation values for quick reference.
      plt.title(
          f"Fit results: mean={mu:.2f}, std={std:.2f}, worst 1% median={np.percentile(data, 99):.2f} worst={np.max(data):.2f}")

      # Identifying the 1% low threshold and marking it on the plot.
      # This represents the value below which 1% of the distances fall, helping identify outliers or unusually small distances.
      percentile_1 = np.percentile(data, 99)
      plt.axvline(percentile_1, color='r', linestyle='dashed', linewidth=2)
      plt.text(percentile_1, max(p)/2, '1% Low', rotation=0, color='red')

      # Labeling axes for clarity.
      plt.xlabel('Distance')
      plt.ylabel('Density')

      # Saving the completed plot to a file.
      filename = "distance_distribution_plot.png"
      plt.savefig(filename)
  else:
      print("No valid distance data to process.")


  distances = []

  for b in examples:
    # Generate embeddings
    b_embed = embed(b)
    distances.append((b_embed, b))
    
  embeddings = [data[0] for data in distances]
  labels = [data[1] for data in distances]

  # Perform UMAP dimensionality reduction
  reducer = umap.UMAP()
  embedding_umap = reducer.fit_transform(embeddings)

  # Create Plotly figure
  fig = make_subplots(rows=1, cols=1)

  # Add scatter plot with labels
  fig.add_trace(go.Scatter(
      x=embedding_umap[:, 0],
      y=embedding_umap[:, 1],
      mode='markers+text',
      marker=dict(color='blue', opacity=0.5),
      text=labels,
      showlegend=False
  ))

  # Update layout
  fig.update_layout(
      title='UMAP Projection of Embeddings with Labels',
      xaxis_title='UMAP Component 1',
      yaxis_title='UMAP Component 2'
  )

  # Save as interactive HTML
  pio.write_html(fig, 'umap_plot_with_labels.html')


  # python3 main.py --dataset word --nt 10000 --nq 10000 --k 1000 --epochs 1200 --embed-dim 100 --save-model

  # --channel 4 --mtc
