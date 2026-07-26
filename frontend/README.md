# frontend

next.js chat interface for the rag assistant. includes a built-in api route that proxies requests to openrouter so the api key stays server-side.

## structure

```
src/
├── app/
│   ├── api/chat/route.js   # server-side api route (rag orchestration + llm streaming)
│   ├── globals.css          # all styles
│   ├── layout.js            # root layout with fonts and metadata
│   └── page.js              # entry point, renders ChatInterface
└── components/
    ├── ChatInterface.jsx    # main chat logic (upload, send, stream)
    ├── ChatBar.jsx          # input bar with file attachment
    ├── MessageList.jsx      # scrollable message container
    ├── MessageBubble.jsx    # individual message with markdown rendering
    └── Sidebar.jsx          # collapsible sidebar with project info
```

## running

```bash
npm install
npm run dev
```

open http://localhost:3000. make sure the fastapi backend is running on port 8000.

## environment

copy `.env.local.example` or create `.env.local` with:

```
OPENROUTER_API_KEY=your-key-here
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```
