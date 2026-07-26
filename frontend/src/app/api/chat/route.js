/**
 * /api/chat — server-side rag orchestration route.
 *
 * handles the full pipeline: receives a question from the frontend,
 * fetches relevant chunks from the fastapi backend, builds the
 * system prompt, and streams the llm response back via sse.
 *
 * the api key stays server-side and never reaches the browser.
 */

import { OpenRouter } from "@openrouter/sdk";

const openrouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
});

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free";

function buildSystemPrompt(chunks) {
  const contextBlocks = chunks
    .map((chunk, i) => {
      const src = chunk.source_file || "unknown";
      const pg = chunk.page_number != null ? `, page ${chunk.page_number}` : "";
      const lec = chunk.lecture_number != null ? `, lecture ${chunk.lecture_number}` : "";
      return `[Source ${i + 1}: ${src}${pg}${lec}]\n${chunk.text}`;
    })
    .join("\n\n---\n\n");

  return `You are a precise, helpful university course assistant. Your role is to answer student questions EXCLUSIVELY based on the course material provided below.

## STRICT RULES
1. ONLY use information from the provided context passages to answer. Do NOT use your general knowledge.
2. If the context does not contain enough information to answer the question, clearly state: "I don't have enough information in the course materials to answer this question."
3. When answering, cite which source(s) you used (e.g., "According to Source 1…").
4. Keep your answers clear, well-structured, and academic in tone.
5. Use markdown formatting (headings, lists, bold, code blocks) to make your answers easy to read.
6. NEVER fabricate information. Accuracy is more important than completeness.

## COURSE CONTEXT
${contextBlocks || "(No context passages were retrieved for this question.)"}`;
}

export async function POST(request) {
  try {
    const { question, fileName } = await request.json();

    if (!question?.trim()) {
      return new Response(JSON.stringify({ error: "question is required." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // fetch relevant chunks from the fastapi backend
    let chunks = [];
    let sources = [];

    try {
      const searchRes = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: question.trim(),
          top_k: 5,
          ...(fileName ? { source_file: fileName } : {}),
        }),
      });

      if (searchRes.ok) {
        const data = await searchRes.json();
        chunks = data.results || [];
        sources = chunks.map((c) => ({
          source_file: c.source_file || "Unknown",
          page_number: c.page_number,
          lecture_number: c.lecture_number,
          score: c.score,
        }));
      }
    } catch (err) {
      console.warn(`backend unreachable: ${err.message}`);
    }

    // build the rag prompt and stream the response
    const systemPrompt = buildSystemPrompt(chunks);
    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: question.trim() },
    ];

    const stream = await openrouter.chat.send({
      chatRequest: { model: MODEL, messages, stream: true },
    });

    const encoder = new TextEncoder();
    const readableStream = new ReadableStream({
      async start(controller) {
        try {
          if (sources.length > 0) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: "sources", sources })}\n\n`)
            );
          }

          for await (const chunk of stream) {
            const content = chunk.choices?.[0]?.delta?.content;
            if (content) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ type: "token", content })}\n\n`)
              );
            }
          }

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "done" })}\n\n`));
        } catch (streamError) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "error", error: streamError.message })}\n\n`)
          );
        } finally {
          controller.close();
        }
      },
    });

    return new Response(readableStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("/api/chat error:", error);
    return new Response(JSON.stringify({ error: error.message || "internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
