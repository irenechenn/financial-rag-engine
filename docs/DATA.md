# Data Notice

## Source

The sample transcript file in this repository was prepared from the Hugging Face dataset:

```text
glopardo/sp500-earnings-transcripts
```

Dataset page:

```text
https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts
```

The upstream dataset card describes the data as S&P 500 earnings call transcripts, primarily covering 2014-2024, with quarterly financial metrics and company fundamentals.

## Citation

The upstream dataset card asks users to cite:

```bibtex
@article{ecb2025genai,
  title={Verba Volant, Transcripta Manent: What Corporate Earnings Calls Reveal About the AI Stock Rally},
  author={Ca' Zorzi, Michele and Lopardo, Gianluigi and Manu, Ana-Simona},
  year=2025,
  institution={European Central Bank},
  number={3093},
  type={Working Paper Series},
  pdf={https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3093~458d28b4bc.en.pdf},
  url={https://glopardo.com/corporatetalks/},
}
```

## Included Sample

This repository includes a small filtered sample for reproducible demos:

```text
data/mini_sp500_transcripts.json
```

The sample was generated with:

```powershell
python scripts/prepare_dataset.py --dataset glopardo/sp500-earnings-transcripts --tickers AAPL MSFT NVDA TSLA GOOGL META AMZN --limit 200
```

## Use Notice

This project uses the included data for educational and demonstration purposes. The original dataset and transcript content remain subject to the upstream dataset terms, source terms, and any rights held by the original transcript providers. If you use or redistribute the data, review the upstream Hugging Face dataset card and associated citation guidance.

