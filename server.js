require('dotenv').config();

// SSL 인증서 검증 완전히 비활성화 (전역 설정)
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const express = require('express');
const path = require('path');
const fs = require('fs');
const auth = require('./src/auth');
const gfid = require('./src/gfidClient');

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.get('/auth/status', async (req, res) => {
	try {
		// Try to obtain a token so status reflects current ability to authenticate
		try {
			await auth.getToken();
		} catch (e) {
			const tokenInfo = await auth.getTokenInfo();
			return res.status(500).json({ ok: false, error: e.message, tokenInfo });
		}

		const tokenInfo = await auth.getTokenInfo();
		res.json({ ok: true, tokenInfo });
	} catch (err) {
		res.status(500).json({ ok: false, error: err.message });
	}
});

app.post('/download', async (req, res) => {
	try {
		const { dsid, returnData } = req.body;
		const dataset = dsid || process.env.DSID;
		if (!dataset) return res.status(400).json({ ok: false, error: 'dsid missing' });

		const outDir = path.resolve(process.cwd(), 'data');
		if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

		// Log single-download start
		if (gfid.logStructured) gfid.logStructured('info', { dsid: dataset, msg: 'single_download_start' });
		const result = await gfid.downloadDataset(dataset, outDir);
		if (gfid.logStructured) gfid.logStructured('info', { dsid: dataset, msg: 'single_download_result', totalFetched: result.totalFetched, pages: result.pageFiles ? result.pageFiles.length : 0 });
		
		// returnData=true인 경우 파일 대신 데이터를 직접 반환
		if (returnData) {
			const allData = [];
			if (result.pageFiles && result.pageFiles.length > 0) {
				for (const pageFile of result.pageFiles) {
					if (fs.existsSync(pageFile)) {
						const fileData = JSON.parse(fs.readFileSync(pageFile, 'utf8'));
						allData.push(...fileData);
						// 파일 읽은 후 삭제 (로컬에 저장하지 않음)
						fs.unlinkSync(pageFile);
					}
				}
			}
			return res.json({ ok: true, result: { data: allData, totalFetched: result.totalFetched } });
		}
		
		res.json({ ok: true, result });
	} catch (err) {
		res.status(500).json({ ok: false, error: err.message });
	}
});

// Bulk download endpoint: POST /download-all
// Body optional: { dsids: ['ds_0101','ds_0202'] }
// If body omitted, will read GFID_DSIDS (comma/newline separated) or GFID_DSIDS_FILE (path to file)
app.post('/download-all', async (req, res) => {
	try {
		// Default ordered DSID list (used if no body/env/file provided)
		const defaultDsids = [
			'ds_0101','ds_0102','ds_0103','ds_0104','ds_0105','ds_0106','ds_0107','ds_0108','ds_0109','ds_0110','ds_0111',
			'ds_0201','ds_0202','ds_0301','ds_0401','ds_0501','ds_0502','ds_0503','ds_0504','ds_0505','ds_0506','ds_0507',
			'ds_0601','ds_0701','ds_0801','ds_0901'
		];

		let dsids = [];
		if (req.body && Array.isArray(req.body.dsids) && req.body.dsids.length > 0) {
			dsids = req.body.dsids;
		} else if (process.env.GFID_DSIDS) {
			// split by comma or newline
			dsids = process.env.GFID_DSIDS.split(/[,\n\r]+/).map(s => s.trim()).filter(Boolean);
		} else if (process.env.GFID_DSIDS_FILE) {
			const filePath = path.resolve(process.cwd(), process.env.GFID_DSIDS_FILE);
			if (fs.existsSync(filePath)) {
				const content = fs.readFileSync(filePath, 'utf8');
				dsids = content.split(/[,\n\r]+/).map(s => s.trim()).filter(Boolean);
			} else {
				return res.status(400).json({ ok: false, error: `GFID_DSIDS_FILE not found: ${filePath}` });
			}
		} else {
			// fallback to built-in ordered list
			dsids = defaultDsids.slice();
		}

		if (dsids.length === 0) return res.status(400).json({ ok: false, error: 'No dsids to process' });

		const outDir = path.resolve(process.cwd(), 'data');
		if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

		const results = [];
		// Log bulk start
		if (gfid.logStructured) gfid.logStructured('info', { msg: 'bulk_download_start', totalDsids: dsids.length });
		const bulkStart = Date.now();
		// Process sequentially to avoid overwhelming the API; change to parallel if desired
		for (let i = 0; i < dsids.length; i++) {
			const dsid = dsids[i];
			try {
				if (gfid.logStructured) gfid.logStructured('info', { msg: 'bulk_download_next', index: i+1, total: dsids.length, dsid });
				const r = await gfid.downloadDataset(dsid, outDir);
				results.push({ dsid, ok: true, result: r });
				if (gfid.logStructured) gfid.logStructured('info', { msg: 'bulk_download_done_one', index: i+1, total: dsids.length, dsid, totalFetched: r.totalFetched, pages: r.pageFiles ? r.pageFiles.length : 0 });
			} catch (e) {
				results.push({ dsid, ok: false, error: e.message });
				if (gfid.logStructured) gfid.logStructured('error', { msg: 'bulk_download_error_one', index: i+1, total: dsids.length, dsid, error: e.message });
			}
		}
		const elapsed = (Date.now() - bulkStart) / 1000;
		if (gfid.logStructured) gfid.logStructured('info', { msg: 'bulk_download_complete', totalDsids: dsids.length, elapsedSec: elapsed });

		res.json({ ok: true, count: results.length, results });
	} catch (err) {
		res.status(500).json({ ok: false, error: err.message });
	}
});

app.get('/', (req, res) => res.send('Influenza local downloader running'));

app.listen(port, () => {
	console.log(`Server listening on http://localhost:${port}`);
});
