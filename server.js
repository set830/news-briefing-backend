require('dotenv').config();
const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3001;

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});
app.use(express.json({ limit: '10mb' }));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const buildPrompt = (filled, tone) => {
  const articlesBlock = filled
    .map((a, i) => `ARTICLE ${i + 1} [Mode: ${a.mode === 'full' ? 'Full Article' : 'Summarize'}]\n${a.text.trim()}`)
    .join('\n\n---\n\n');

  const toneInstruction =
    tone === 'conversational'
      ? 'CONVERSATIONAL — write summaries the way a friend would tell another friend about the story: fun, interesting, informal, with some personality. Do not overdo it.'
      : 'DIRECT — write summaries that match and preserve the tone and voice of the original article.';

  return `You are producing a personal audio news briefing to be read aloud by a text-to-speech voice. Follow all instructions precisely.

TONE SETTING: ${toneInstruction}

RULES FOR "Summarize" ARTICLES:
- Keep the original headline exactly as written, spoken naturally (e.g. "The headline is: ...")
- Include the author's name ONLY if the piece is clearly an op-ed or opinion column. Otherwise omit it entirely.
- Write one paragraph summary in the chosen tone.
- Follow the summary with 3 to 4 key points. Introduce them naturally for audio: "A few key points: First, ... Second, ... Third, ..." — do NOT use bullet symbols or dashes.
- If the article includes a specific example, anecdote, or story that meaningfully illustrates one of the key points, weave it in naturally after that point. Only include it if it genuinely helps the listener understand the concept — skip it if it is tangential or redundant.

RULES FOR "Full Article" ARTICLES:
- Remove everything that is not the article body: ads, related article links, newsletter prompts, comment sections, social sharing prompts, author bios at the bottom, and any other website chrome.
- Keep the clean article prose, preserving its paragraphs.

SCRIPT FORMAT:
- Open with a single short intro line such as: "Here's your morning briefing."
- After each article (except the last), add one brief natural spoken transition before the next article — vary these so they do not all sound identical.
- Close with one brief sign-off line.
- Write entirely in plain spoken prose. No markdown. No asterisks, hashes, or symbols. No bullet points. Numbers are fine.
- The script will be fed directly to a TTS engine, so every character will be spoken. Write accordingly.

ARTICLE BREAK MARKERS:
- Immediately before the content of each article begins (after the intro line, and after any transition), insert a marker on its own line in the exact format: <<ARTICLE_N>> where N is the article number (1, 2, 3, etc.)
- Do NOT place a marker before the opening intro line.
- The marker must appear on its own line immediately before that article's headline or first sentence.
- These markers will be stripped before TTS processing — do not worry about them being read aloud.

HERE ARE THE ARTICLES:

${articlesBlock}

Write the complete briefing script now:`;
};

const parseScript = (rawScript) => {
  const markerRegex = /<<ARTICLE_(\d+)>>\n?/g;
  const allMarkers = [];
  let m;
  while ((m = markerRegex.exec(rawScript)) !== null) {
    allMarkers.push({ fullMatch: m[0], pos: m.index, num: parseInt(m[1]) });
  }

  const script = rawScript.replace(/<<ARTICLE_\d+>>\n?/g, '');
  const totalLen = script.length;

  let cumulativeRemoved = 0;
  const articleBreaks = totalLen === 0 ? [] : allMarkers.map(marker => {
    const adjustedPos = marker.pos - cumulativeRemoved;
    cumulativeRemoved += marker.fullMatch.length;
    return { n: marker.num, frac: Math.min(1, adjustedPos / totalLen) };
  });

  return { script, articleBreaks };
};

const chunkText = (text, maxChars = 7000) => {
  if (text.length <= maxChars) return [text];
  const chunks = [];
  let remaining = text;
  while (remaining.length > maxChars) {
    let splitAt = remaining.lastIndexOf('. ', maxChars);
    if (splitAt === -1) splitAt = remaining.lastIndexOf(' ', maxChars);
    if (splitAt === -1) splitAt = maxChars;
    else splitAt += 1;
    chunks.push(remaining.slice(0, splitAt).trim());
    remaining = remaining.slice(splitAt).trim();
  }
  if (remaining.length > 0) chunks.push(remaining);
  return chunks;
};

const generateAudioFromScript = async (script) => {
  const chunks = chunkText(script);
  const audioBuffers = await Promise.all(chunks.map(chunk =>
    openai.audio.speech.create({
      model: 'gpt-4o-mini-tts',
      voice: 'ash',
      input: chunk,
      response_format: 'mp3',
    }).then(r => r.arrayBuffer()).then(b => Buffer.from(b))
  ));
  return Buffer.concat(audioBuffers).toString('base64');
};

const streamAudioFromScript = async (script, onChunk) => {
  const chunks = chunkText(script);
  await Promise.all(chunks.map((chunk, i) =>
    openai.audio.speech.create({
      model: 'gpt-4o-mini-tts',
      voice: 'ash',
      input: chunk,
      response_format: 'mp3',
    }).then(r => r.arrayBuffer()).then(b => {
      onChunk(Buffer.from(b).toString('base64'), i, chunks.length);
    })
  ));
};


// Step 1: Generate script with Claude
app.post('/api/generate-script', async (req, res) => {
  const { articles, tone } = req.body;

  if (!articles || !Array.isArray(articles)) {
    return res.status(400).json({ error: 'Invalid request: articles must be an array.' });
  }

  const filled = articles.filter(a => a.text && a.text.trim().length > 0);
  if (filled.length === 0) {
    return res.status(400).json({ error: 'No articles provided.' });
  }

  try {
    const prompt = buildPrompt(filled, tone);
    const message = await anthropic.messages.create({
      model: 'claude-sonnet-5-5',
      max_tokens: 8192,
      messages: [{ role: 'user', content: prompt }],
    });

    const textBlock = message.content.find(b => b.type === 'text');
    if (!textBlock) throw new Error('No text content in Claude response');
    const { script, articleBreaks } = parseScript(textBlock.text);
    console.log(`Script generated (${script.length} chars), ${articleBreaks.length} article breaks.`);
    res.json({ script, articleBreaks });
  } catch (err) {
    console.error('Script error:', err.message);
    res.status(500).json({ error: err.message || 'Internal server error' });
  }
});

// Step 2: Generate audio from script — parallel TTS, chunked streaming response
app.post('/api/generate-audio', async (req, res) => {
  const { script } = req.body;

  if (!script || typeof script !== 'string' || script.trim().length === 0) {
    return res.status(400).json({ error: 'Invalid request: script must be a non-empty string.' });
  }

  try {
    const chunks = chunkText(script);

    // Kick off all TTS calls in parallel
    const promises = chunks.map(chunk =>
      openai.audio.speech.create({
        model: 'gpt-4o-mini-tts',
        voice: 'ash',
        input: chunk,
        response_format: 'mp3',
      }).then(r => r.arrayBuffer()).then(b => Buffer.from(b))
    );

    res.set({ 'Content-Type': 'audio/mpeg', 'Transfer-Encoding': 'chunked', 'Cache-Control': 'no-cache' });
    res.flushHeaders(); // send headers immediately so Railway doesn't time out before first chunk

    // Await and stream each chunk in order — keeps connection alive during generation
    for (const promise of promises) {
      const buf = await promise;
      res.write(buf);
    }
    res.end();
    console.log(`Audio streamed successfully (${chunks.length} chunk(s)).`);
  } catch (err) {
    console.error('Audio error:', err.message);
    if (!res.headersSent) {
      res.status(500).json({ error: err.message || 'Internal server error' });
    }
  }
});

// Legacy combined endpoint (kept for local dev convenience)
app.post('/api/generate', async (req, res) => {
  const { articles, tone } = req.body;

  if (!articles || !Array.isArray(articles)) {
    return res.status(400).json({ error: 'Invalid request: articles must be an array.' });
  }

  const filled = articles.filter(a => a.text && a.text.trim().length > 0);
  if (filled.length === 0) {
    return res.status(400).json({ error: 'No articles provided.' });
  }

  try {
    const prompt = buildPrompt(filled, tone);
    const message = await anthropic.messages.create({
      model: 'claude-sonnet-5-5',
      max_tokens: 8192,
      messages: [{ role: 'user', content: prompt }],
    });

    const textBlock = message.content.find(b => b.type === 'text');
    if (!textBlock) throw new Error('No text content in Claude response');
    const { script, articleBreaks } = parseScript(textBlock.text);
    console.log(`Script generated (${script.length} chars), ${articleBreaks.length} article breaks. Sending to OpenAI TTS...`);

    const audioBase64 = await generateAudioFromScript(script);
    console.log('Audio generated successfully.');

    res.json({ script, audio: audioBase64, audioType: 'audio/mpeg', articleBreaks });
  } catch (err) {
    const detail = err.response?.data
      ? Buffer.isBuffer(err.response.data)
        ? err.response.data.toString()
        : JSON.stringify(err.response.data)
      : err.message;
    console.error('Error:', detail);
    res.status(500).json({ error: detail || 'Internal server error' });
  }
});

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
