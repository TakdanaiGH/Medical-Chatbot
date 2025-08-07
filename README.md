# Medical-Chatbot
Medical Chatbot using Retrieval-Augmented Generation (RAG) data from Agnos.

---

## Hardware Recommendations
I recommend using a GPU with CUDA and at least 40 GB of VRAM (I run it on an NVIDIA A100). You can adjust GPU utilization and change models if necessary in the serve scripts.

---
## Demo

Watch the demo video to see the Medical Chatbot in action: [View Demo](https://drive.google.com/file/d/1CP_nqMJSzqXyVDkMJNB9vjU0lNMNSsGX/view?usp=sharing)

---
## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/TakdanaiGH/Medical-Chatbot.git
cd Medical-Chatbot
```

### Create and Activate a New Conda Environment
```bash
conda create --name med-env python=3.10
conda activate med-env
```

### Install Required Python Libraries
```bash
pip install -r requirements.txt
```

**Important:** Don't forget to rename your environment configuration file to `.env` and configure it properly.

---

## Usage

### Auto Scrape RAG Data from Agnos Threads
The scraper collects data from [Agnos Health forums](https://www.agnoshealth.com/forums). It will:
- Scrape all pages, skipping duplicate thread IDs
- Pause for 5 minutes after finishing a full cycle
- Can be run in the background

**Alternative:** You can download pre-scraped data from [this link](https://drive.google.com/file/d/1CP_nqMJSzqXyVDkMJNB9vjU0lNMNSsGX/view?usp=sharing) to test the system without running the scraper.

Run the scraper:
```bash
./run_scraper.sh
```
Logs are saved to `scraper.log`.

---

## Deployment

To deploy the website, you need to serve 2 models sequentially:

### 1. Serve the LLM Model
```bash
nohup ./run_chat_model.sh &
```

**⚠️ Important Notes:**
- This will take a while to load
- You **must** wait until it's ready before serving the next model
- Monitor progress in the log file: `vllm_chat.log`

### 2. Serve the Embedding Model (for RAG)
After the LLM model is fully loaded, serve the embedding model:
```bash
nohup ./run_embed_model.sh &
```

### 3. Start Streamlit Interface
Once both models are running, start the Streamlit web interface:
```bash
nohup ./run_streamlit.sh &
```

The Streamlit app will run on port 8501 by default. You can find the exact port in the Streamlit log file.

### 4. External Access (Optional)
If you need external access to your chatbot, use ngrok:
```bash
ngrok http 8501
```

---

## Script Structure
```
Medical-Chatbot/
├── run_scraper.sh          # Data scraping script
├── run_chat_model.sh       # LLM model server
├── run_embed_model.sh      # Embedding model server
├── run_streamlit.sh        # Web interface launcher
└── .env                   # Environment configuration
```

---

## Troubleshooting

1. **Models not loading:** Ensure you have sufficient GPU memory (40GB+ recommended)
2. **Port conflicts:** Check if ports are already in use
3. **Environment issues:** Make sure your `.env` file is properly configured
4. **Sequential loading:** Always wait for the LLM model to fully load before starting the embedding model

---
