// const fs = require('fs');
// const path = require('path');
// const axios = require('axios');
// const auth = require('./auth');

// // Configurable via env vars:
// // GFID_API_BASE - base URL (required for real use)
// // GFID_API_PATH - path template, use {dsid} for dataset id (default: /datasets/{dsid})
// // GFID_PAGE_MODE - "page" or "offset" (default: page)
// // GFID_PAGE_PARAM - page query param name (default: page)
// // GFID_SIZE_PARAM - size/limit query param name (default: size)
// // GFID_ITEMS_KEY - path to array in response: e.g. items,data or empty for top-level array
// // GFID_PAGE_SIZE - page size (default 100)
// const GFID_API_BASE = (process.env.GFID_API_BASE || '').replace(/\/$/, '');
// const GFID_API_PATH = process.env.GFID_API_PATH || '/datasets/{dsid}';
// const GFID_PAGE_MODE = (process.env.GFID_PAGE_MODE || 'page').toLowerCase();
// const GFID_PAGE_PARAM = process.env.GFID_PAGE_PARAM || 'page';
// const GFID_SIZE_PARAM = process.env.GFID_SIZE_PARAM || 'size';
// const GFID_ITEMS_KEY = process.env.GFID_ITEMS_KEY || ''; // e.g. 'items' or 'data' or ''
// const GFID_PAGE_SIZE = parseInt(process.env.GFID_PAGE_SIZE || process.env.cnt || '100', 10);
// const FROM = process.env.FROM;
// const TO = process.env.TO;
// // By default do NOT merge per-page files into a single file. Set to 'true' to enable.
// const GFID_MERGE_PAGES = (process.env.GFID_MERGE_PAGES || 'false').toLowerCase() === 'true';

// if (!GFID_API_BASE) {
//   // Structured warning for missing configuration
//   console.warn(JSON.stringify({
//     ts: new Date().toISOString(),
//     level: 'warn',
//     msg: 'GFID_API_BASE is not set — client will write stub data until configured. Set GFID_API_BASE in .env to enable real downloads.'
//   }));
// }

// function logStructured(level, info) {
//   const out = Object.assign({ ts: new Date().toISOString(), level }, info || {});
//   try {
//     console.log(JSON.stringify(out));
//   } catch (e) {
//     // fallback to plain logging
//     console.log(out);
//   }
// }

// async function requestWithRetry(url, config, attempts = 3) {
//   let attempt = 0;
//   while (attempt < attempts) {
//     try {
//       return await axios.get(url, config);
//     } catch (err) {
//       attempt++;
//       const canRetry = err.code === 'ECONNABORTED' || err.code === 'ECONNREFUSED' || (err.response && err.response.status >= 500);
//       if (!canRetry || attempt >= attempts) throw err;
//       await new Promise(r => setTimeout(r, 500 * Math.pow(2, attempt)));
//     }
//   }
// }

// function resolveItems(respData) {
//   if (!respData) return [];
//   if (GFID_ITEMS_KEY) {
//     const parts = GFID_ITEMS_KEY.split('.');
//     let cur = respData;
//     for (const p of parts) {
//       if (cur == null) return [];
//       cur = cur[p];
//     }
//     return Array.isArray(cur) ? cur : [];
//   }
//   if (Array.isArray(respData)) return respData;
//   if (Array.isArray(respData.items)) return respData.items;
//   if (Array.isArray(respData.data)) return respData.data;
//   return [];
// }

// async function downloadDataset(dsid, outDir) {
//   const startTime = Date.now();
//   let pagesFetched = 0;
//   logStructured('info', { dsid, msg: 'download_start', startTime: new Date(startTime).toISOString() });
//   // If GFID_API_BASE not configured, keep stub behaviour
//   if (!GFID_API_BASE) {
//     // Write a sample page file so caller can see per-page output even in stub mode
//     if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
//     const sampleItems = [{ dsid, sample: true, downloadedAt: new Date().toISOString() }];
//     const pageFile = path.join(outDir, `${dsid}_page_1.json`);
//     fs.writeFileSync(pageFile, JSON.stringify(sampleItems, null, 2));
//     // complete log for stub run
//     logStructured('info', { dsid, msg: 'download_complete', pagesFetched: 1, totalFetched: sampleItems.length, elapsedSec: (Date.now()-startTime)/1000 });
//     return { combinedPath: null, pageFiles: [pageFile], totalFetched: sampleItems.length };
//   }

//   const token = await auth.getToken();
//   const headers = { Authorization: `Bearer ${token}` };

//   if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
//   const combinedPath = path.join(outDir, `${dsid}.json`);
//   const pageFiles = [];
//   let totalFetched = 0;

//   // paging state
//   let page = 1;
//   let offset = 0;
//   // Keep last page payload to detect repeating pages from the API
//   let prevItemsJson = null;

//   const pathContainsFromOrTo = GFID_API_PATH.includes('{from}') || GFID_API_PATH.includes('{to}');
//   while (true) {
//     // Resolve placeholders in path: {dsid}, and optionally {from}/{to}
//     let pathResolved = GFID_API_PATH.replace('{dsid}', encodeURIComponent(dsid));
//     if (pathResolved.includes('{from}')) {
//       pathResolved = pathResolved.replace(/{from}/g, encodeURIComponent(FROM || ''));
//     }
//     if (pathResolved.includes('{to}')) {
//       pathResolved = pathResolved.replace(/{to}/g, encodeURIComponent(TO || ''));
//     }
//     const url = `${GFID_API_BASE}${pathResolved}`;
//     const params = new URLSearchParams();
//     if (GFID_PAGE_MODE === 'offset') {
//       params.append(GFID_PAGE_PARAM, String(offset));
//       params.append(GFID_SIZE_PARAM, String(GFID_PAGE_SIZE));
//     } else {
//       params.append(GFID_PAGE_PARAM, String(page));
//       params.append(GFID_SIZE_PARAM, String(GFID_PAGE_SIZE));
//     }
//     // Only add from/to as query params when the path template does not already include them
//     if (!pathContainsFromOrTo) {
//       if (FROM) params.append('from', FROM);
//       if (TO) params.append('to', TO);
//     }

//     const fullUrl = `${url}?${params.toString()}`;
//     // Structured request log
//     logStructured('info', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, url: fullUrl });
//     let resp;
//     try {
//       resp = await requestWithRetry(fullUrl, { headers, timeout: 15000 }, 3);
//     } catch (err) {
//       if (err.response) {
//         const body = err.response.data;
//         throw new Error(`GFID request failed: ${err.response.status} ${JSON.stringify(body)}`);
//       }
//       throw new Error('GFID request failed: ' + (err.message || 'unknown'));
//     }

//     const items = resolveItems(resp.data);
//     // Detect identical pages returned repeatedly and stop to avoid endless duplicates
//     try {
//       const itemsJson = JSON.stringify(items);
//       if (prevItemsJson && itemsJson === prevItemsJson) {
//         logStructured('warn', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, url: fullUrl, msg: 'same page content received — stopping to avoid duplicates' });
//         break;
//       }
//       prevItemsJson = itemsJson;
//     } catch (e) {
//       // If stringify fails for some reason, ignore and continue
//     }
//     if (!items || items.length === 0) break;

//     const pageFile = path.join(outDir, `${dsid}_page_${page}.json`);
//     fs.writeFileSync(pageFile, JSON.stringify(items, null, 2));
//     // update counters
//     pagesFetched += 1;
//     totalFetched += items.length;
//     pageFiles.push(pageFile);
//     // Log page-level progress (pages fetched, items this page, total so far)
//     logStructured('info', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, pageFile, itemsThisPage: items.length, pagesFetched, totalFetched, elapsedSec: (Date.now()-startTime)/1000 });

//     // Stop conditions
//     if (items.length < GFID_PAGE_SIZE) break;
//     // If using offset mode, advance offset
//     if (GFID_PAGE_MODE === 'offset') {
//       offset += GFID_PAGE_SIZE;
//       // small safety to prevent infinite loops
//       if (offset > 10000000) break;
//     } else {
//       page++;
//       if (page > 100000) break; // safety
//     }
//   }

//   // Optionally combine pages into a single file. Default behavior is to keep per-page files.
//   let finalCombinedPath = null;
//   if (GFID_MERGE_PAGES && pageFiles.length > 0) {
//     const combinedStream = fs.createWriteStream(combinedPath, { flags: 'w' });
//     combinedStream.write('[');
//     let first = true;
//     for (const pf of pageFiles) {
//       const arr = JSON.parse(fs.readFileSync(pf, 'utf8'));
//       for (const it of arr) {
//         if (!first) combinedStream.write(',\n');
//         combinedStream.write(JSON.stringify(it));
//         first = false;
//       }
//     }
//     combinedStream.write(']\n');
//     combinedStream.end();
//     finalCombinedPath = combinedPath;
//     logStructured('info', { dsid, mergedPath: finalCombinedPath, pageFilesCount: pageFiles.length });
//   }

//   // final completion log
//   logStructured('info', { dsid, msg: 'download_complete', pagesFetched, totalFetched, combinedPath: finalCombinedPath, elapsedSec: (Date.now()-startTime)/1000 });

//   return { combinedPath: finalCombinedPath, pageFiles, totalFetched };
// }

// module.exports = { downloadDataset, logStructured };

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const https = require('https');
const auth = require('./auth');

// SSL 인증서 검증 비활성화
const httpsAgent = new https.Agent({
	rejectUnauthorized: false
});

// Configurable via env vars:
// GFID_API_BASE - base URL (required for real use)
// GFID_API_PATH - path template, use {dsid} for dataset id (default: /datasets/{dsid})
// GFID_PAGE_MODE - "page" or "offset" (default: page)
// GFID_PAGE_PARAM - page query param name (default: page)
// GFID_SIZE_PARAM - size/limit query param name (default: size)
// GFID_ITEMS_KEY - path to array in response: e.g. items,data or empty for top-level array
// GFID_PAGE_SIZE - page size (default 100)
const GFID_API_BASE = (process.env.GFID_API_BASE || '').replace(/\/$/, '');
const GFID_API_PATH = process.env.GFID_API_PATH || '/datasets/{dsid}';
const GFID_PAGE_MODE = (process.env.GFID_PAGE_MODE || 'page').toLowerCase();
const GFID_PAGE_PARAM = process.env.GFID_PAGE_PARAM || 'page';
const GFID_SIZE_PARAM = process.env.GFID_SIZE_PARAM || 'size';
const GFID_ITEMS_KEY = process.env.GFID_ITEMS_KEY || ''; // e.g. 'items' or 'data' or ''
const GFID_PAGE_SIZE = parseInt(process.env.GFID_PAGE_SIZE || process.env.cnt || '100', 10);
const FROM = process.env.FROM;
const TO = process.env.TO;
// By default do NOT merge per-page files into a single file. Set to 'true' to enable.
const GFID_MERGE_PAGES = (process.env.GFID_MERGE_PAGES || 'false').toLowerCase() === 'true';

if (!GFID_API_BASE) {
  // Structured warning for missing configuration
  console.warn(JSON.stringify({
    ts: new Date().toISOString(),
    level: 'warn',
    msg: 'GFID_API_BASE is not set — client will write stub data until configured. Set GFID_API_BASE in .env to enable real downloads.'
  }));
}

function logStructured(level, info) {
  const out = Object.assign({ ts: new Date().toISOString(), level }, info || {});
  try {
    console.log(JSON.stringify(out));
  } catch (e) {
    // fallback to plain logging
    console.log(out);
  }
}

async function requestWithRetry(url, config, attempts = 3) {
  let attempt = 0;
  while (attempt < attempts) {
    try {
      // SSL 검증 비활성화 추가
      const finalConfig = { ...config, httpsAgent };
      return await axios.get(url, finalConfig);
    } catch (err) {
      attempt++;
      const canRetry = err.code === 'ECONNABORTED' || err.code === 'ECONNREFUSED' || (err.response && err.response.status >= 500);
      if (!canRetry || attempt >= attempts) throw err;
      await new Promise(r => setTimeout(r, 500 * Math.pow(2, attempt)));
    }
  }
}

function resolveItems(respData) {
  if (!respData) return [];
  if (GFID_ITEMS_KEY) {
    const parts = GFID_ITEMS_KEY.split('.');
    let cur = respData;
    for (const p of parts) {
      if (cur == null) return [];
      cur = cur[p];
    }
    return Array.isArray(cur) ? cur : [];
  }
  if (Array.isArray(respData)) return respData;
  if (Array.isArray(respData.items)) return respData.items;
  if (Array.isArray(respData.data)) return respData.data;
  return [];
}

async function downloadDataset(dsid, outDir) {
  const startTime = Date.now();
  let pagesFetched = 0;
  logStructured('info', { dsid, msg: 'download_start', startTime: new Date(startTime).toISOString() });
  // If GFID_API_BASE not configured, keep stub behaviour
  if (!GFID_API_BASE) {
    // Write a sample page file so caller can see per-page output even in stub mode
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    const sampleItems = [{ dsid, sample: true, downloadedAt: new Date().toISOString() }];
    const pageFile = path.join(outDir, `${dsid}_page_1.json`);
    fs.writeFileSync(pageFile, JSON.stringify(sampleItems, null, 2));
    // complete log for stub run
    logStructured('info', { dsid, msg: 'download_complete', pagesFetched: 1, totalFetched: sampleItems.length, elapsedSec: (Date.now()-startTime)/1000 });
    return { combinedPath: null, pageFiles: [pageFile], totalFetched: sampleItems.length };
  }

  const token = await auth.getToken();
  const headers = { Authorization: `Bearer ${token}` };

  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const combinedPath = path.join(outDir, `${dsid}.json`);
  const pageFiles = [];
  let totalFetched = 0;

  // paging state
  let page = 1;
  let offset = 0;
  // Keep last page payload to detect repeating pages from the API
  let prevItemsJson = null;

  const pathContainsFromOrTo = GFID_API_PATH.includes('{from}') || GFID_API_PATH.includes('{to}');
  while (true) {
    // Resolve placeholders in path: {dsid}, and optionally {from}/{to}
    let pathResolved = GFID_API_PATH.replace('{dsid}', encodeURIComponent(dsid));
    if (pathResolved.includes('{from}')) {
      pathResolved = pathResolved.replace(/{from}/g, encodeURIComponent(FROM || ''));
    }
    if (pathResolved.includes('{to}')) {
      pathResolved = pathResolved.replace(/{to}/g, encodeURIComponent(TO || ''));
    }
    const url = `${GFID_API_BASE}${pathResolved}`;
    const params = new URLSearchParams();
    if (GFID_PAGE_MODE === 'offset') {
      params.append(GFID_PAGE_PARAM, String(offset));
      params.append(GFID_SIZE_PARAM, String(GFID_PAGE_SIZE));
    } else {
      params.append(GFID_PAGE_PARAM, String(page));
      params.append(GFID_SIZE_PARAM, String(GFID_PAGE_SIZE));
    }
    // Only add from/to as query params when the path template does not already include them
    if (!pathContainsFromOrTo) {
      if (FROM) params.append('from', FROM);
      if (TO) params.append('to', TO);
    }

    const fullUrl = `${url}?${params.toString()}`;
    // Structured request log
    logStructured('info', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, url: fullUrl });
    let resp;
    try {
      resp = await requestWithRetry(fullUrl, { headers, timeout: 15000 }, 3);
    } catch (err) {
      if (err.response) {
        const body = err.response.data;
        throw new Error(`GFID request failed: ${err.response.status} ${JSON.stringify(body)}`);
      }
      throw new Error('GFID request failed: ' + (err.message || 'unknown'));
    }

    const items = resolveItems(resp.data);
    // Detect identical pages returned repeatedly and stop to avoid endless duplicates
    try {
      const itemsJson = JSON.stringify(items);
      if (prevItemsJson && itemsJson === prevItemsJson) {
        logStructured('warn', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, url: fullUrl, msg: 'same page content received — stopping to avoid duplicates' });
        break;
      }
      prevItemsJson = itemsJson;
    } catch (e) {
      // If stringify fails for some reason, ignore and continue
    }
    if (!items || items.length === 0) break;

    const pageFile = path.join(outDir, `${dsid}_page_${page}.json`);
    fs.writeFileSync(pageFile, JSON.stringify(items, null, 2));
    // update counters
    pagesFetched += 1;
    totalFetched += items.length;
    pageFiles.push(pageFile);
    // Log page-level progress (pages fetched, items this page, total so far)
    logStructured('info', { dsid, page: GFID_PAGE_MODE === 'offset' ? offset : page, pageFile, itemsThisPage: items.length, pagesFetched, totalFetched, elapsedSec: (Date.now()-startTime)/1000 });

    // Stop conditions
    if (items.length < GFID_PAGE_SIZE) break;
    // If using offset mode, advance offset
    if (GFID_PAGE_MODE === 'offset') {
      offset += GFID_PAGE_SIZE;
      // small safety to prevent infinite loops
      if (offset > 10000000) break;
    } else {
      page++;
      if (page > 100000) break; // safety
    }
  }

  // Optionally combine pages into a single file. Default behavior is to keep per-page files.
  let finalCombinedPath = null;
  if (GFID_MERGE_PAGES && pageFiles.length > 0) {
    const combinedStream = fs.createWriteStream(combinedPath, { flags: 'w' });
    combinedStream.write('[');
    let first = true;
    for (const pf of pageFiles) {
      const arr = JSON.parse(fs.readFileSync(pf, 'utf8'));
      for (const it of arr) {
        if (!first) combinedStream.write(',\n');
        combinedStream.write(JSON.stringify(it));
        first = false;
      }
    }
    combinedStream.write(']\n');
    combinedStream.end();
    finalCombinedPath = combinedPath;
    logStructured('info', { dsid, mergedPath: finalCombinedPath, pageFilesCount: pageFiles.length });
  }

  // final completion log
  logStructured('info', { dsid, msg: 'download_complete', pagesFetched, totalFetched, combinedPath: finalCombinedPath, elapsedSec: (Date.now()-startTime)/1000 });

  return { combinedPath: finalCombinedPath, pageFiles, totalFetched };
}

module.exports = { downloadDataset, logStructured };

