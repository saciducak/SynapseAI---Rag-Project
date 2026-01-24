# 🚀 Deployment Guide for SynapseAI

Since this is a full-stack application (FastAPI Backend + HTML/JS Frontend), we will deploy them separately for the best performance and scalability.

## 1. GitHub Repository (COMPLETED ✅)
The code has been successfully pushed to:
[https://github.com/saciducak/SynapseAI---Rag-Project](https://github.com/saciducak/SynapseAI---Rag-Project)

✅ Code is live.
✅ Branches are set.

You can skip step 1 and proceed directly to deployment.

---

## 2. Deploy Backend (Render.com)
*Free and easy for Python/FastAPI*

1. Sign up at [render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repo `synapse-ai`.
4. Configure:
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**.
6. **Copy your Backend URL** (e.g., `https://synapse-backend.onrender.com`).

---

## 3. Deploy Frontend (Vercel)
*Best for static sites and Next.js*

1. Sign up at [vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import `synapse-ai`.
4. Configure:
   - **Framework Preset**: `Other` (since we are using a custom `index.html`) or `Next.js` if you revert to the full app.
   - **Root Directory**: `frontend`
5. Click **Deploy**.

### ⚠️ IMPORTANT: Connect Frontend to Backend

By default, the frontend tries to connect to `localhost:8002`. You need to update this to your new Render Backend URL.

1. Open `frontend/index.html` locally.
2. Find this line (search for `API`):
   ```javascript
   const API = 'http://localhost:8002/api';
   ```
3. Change it to your Render URL:
   ```javascript
   const API = 'https://YOUR-BACKEND-URL.onrender.com/api';
   ```
4. Commit and push:
   ```bash
   git add frontend/index.html
   git commit -m "Update API URL for production"
   git push
   ```
5. Vercel will automatically redeploy!

---

## 4. Environment Variables (Optional)
If you use OpenAI keys in the backend, add them in Render:
1. Go to Render Dashboard -> **Environment**.
2. Add `OPENAI_API_KEY`, `GROQ_API_KEY`, etc.
