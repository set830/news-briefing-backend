require('dotenv').config();
const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

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
    // Build Claude prompt
    const articlesBlock = filled
      .map((a, i) => `ARTICLE ${i + 1} [Mode: ${a.mode === 'full' ? 'Full Article' : 'Summarize'}]\n${a.text.trim()}`)
      .join('\n\n---\n\n');

    const toneInstruction =
      tone === 'conversational'
        ? 'CONVERSATIONAL — write summaries the way a friend would tell another friend about the story: fun, interesting, informal, with some personality. Do not overdo it.'
        : 'DIRECT — write summaries that match and preserve the tone and voice of the original article.';

    const prompt = `You are producing a personal audio news briefing to be read aloud by a text-to-speech voice. Follow all instructions precisely.

TONE SETTING: ${toneInstruction}

RULES FOR "Summarize" ARTICLES:
- Keep the original headline exactly as written, spoken naturally (e.g. "The headline is: ...")
- Include the author's name ONLY if the piece is clearly an op-ed or opinion column. Otherwise omit it entirely.
- Write one paragraph summary in the chosen tone.
- Follow the summary with 3 to 4 key points. Introduce them naturally for audio: "A few key points: First, ... Second, ... Third, ..." — do NOT use bullet symbols or dashes.

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

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 8192,
      messages: [{ role: 'user', content: prompt }],
    });

    const rawScript = message.content[0].text;

    // Extract article break positions before stripping markers
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

    console.log(`Script generated (${script.length} chars), ${articleBreaks.length} article breaks. Sending to ElevenLabs...`);

    // Call ElevenLabs
    const elResponse = await axios.post(
      'https://api.elevenlabs.io/v1/text-to-speech/wWWn96OtTHu1sn8SRGEr',
      {
        text: script,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      },
      {
        headers: {
          'xi-api-key': process.env.ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
          Accept: 'audio/mpeg',
        },
        responseType: 'arraybuffer',
        timeout: 120000,
      }
    );

    const audioBase64 = Buffer.from(elResponse.data).toString('base64');
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
