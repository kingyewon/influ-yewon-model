// const axios = require('axios');

// const {
// 	SERVER_URL,
// 	REALM,
// 	CLIENT_ID,
// 	CLIENT_SECRET
// } = process.env;

// if (!SERVER_URL || !REALM || !CLIENT_ID) {
// 	console.warn('Missing Keycloak env vars. Check .env or .env.example');
// }

// let cached = {
// 	access_token: null,
// 	expires_at: 0
// };

// async function fetchToken() {
// 	const tokenUrl = `${SERVER_URL.replace(/\/$/, '')}/realms/${REALM}/protocol/openid-connect/token`;
// 	const params = new URLSearchParams();
// 	params.append('grant_type', 'client_credentials');
// 	params.append('client_id', CLIENT_ID);
// 	if (CLIENT_SECRET) params.append('client_secret', CLIENT_SECRET);

// 	try {
// 		const resp = await axios.post(tokenUrl, params.toString(), {
// 			headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
// 			timeout: 60000,  // 60초로 증가
// 			httpsAgent: new (require('https').Agent)({  
// 				rejectUnauthorized: false  // SSL 인증서 검증 비활성화
// 			})
// 		});

// 		const data = resp.data;
// 		const now = Math.floor(Date.now() / 1000);
// 		cached.access_token = data.access_token;
// 		cached.expires_at = now + (data.expires_in || 300);
// 		return cached;
// 	} catch (err) {
// 		// Provide helpful debug info while avoiding sensitive data leaks
// 		if (err.response) {
// 			console.error('Keycloak token fetch failed:', err.response.status, err.response.data && JSON.stringify(err.response.data));
// 			throw new Error(`Keycloak token request failed with status ${err.response.status}`);
// 		} else if (err.request) {
// 			console.error('Keycloak token fetch no response:', err.message);
// 			throw new Error('No response from Keycloak token endpoint');
// 		} else {
// 			console.error('Keycloak token fetch error:', err.message);
// 			throw new Error('Failed to request Keycloak token: ' + err.message);
// 		}
// 	}
// }

// async function getToken() {
// 	const now = Math.floor(Date.now() / 1000);
// 	// refresh 30s before expiry
// 	if (cached.access_token && cached.expires_at - 30 > now) {
// 		return cached.access_token;
// 	}
// 	await fetchToken();
// 	return cached.access_token;
// }

// async function getTokenInfo() {
// 	const now = Math.floor(Date.now() / 1000);
// 	return {
// 		hasToken: !!cached.access_token,
// 		expiresAt: cached.expires_at,
// 		secondsUntilExpiry: Math.max(0, cached.expires_at - now)
// 	};
// }

// module.exports = { getToken, getTokenInfo };


const axios = require('axios');
const https = require('https');

const {
	SERVER_URL,
	REALM,
	CLIENT_ID,
	CLIENT_SECRET
} = process.env;

if (!SERVER_URL || !REALM || !CLIENT_ID) {
	console.warn('Missing Keycloak env vars. Check .env or .env.example');
}

let cached = {
	access_token: null,
	expires_at: 0
};

// SSL 인증서 검증을 비활성화하는 HTTPS Agent 생성
const httpsAgent = new https.Agent({
	rejectUnauthorized: false
});

async function fetchToken() {
	const tokenUrl = `${SERVER_URL.replace(/\/$/, '')}/realms/${REALM}/protocol/openid-connect/token`;
	const params = new URLSearchParams();
	params.append('grant_type', 'client_credentials');
	params.append('client_id', CLIENT_ID);
	if (CLIENT_SECRET) params.append('client_secret', CLIENT_SECRET);

	try {
		const resp = await axios.post(tokenUrl, params.toString(), {
			headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
			timeout: 60000,  // 60초 타임아웃
			httpsAgent: httpsAgent  // SSL 인증서 검증 비활성화
		});

		const data = resp.data;
		const now = Math.floor(Date.now() / 1000);
		cached.access_token = data.access_token;
		cached.expires_at = now + (data.expires_in || 300);
		return cached;
	} catch (err) {
		// Provide helpful debug info while avoiding sensitive data leaks
		if (err.response) {
			console.error('Keycloak token fetch failed:', err.response.status, err.response.data && JSON.stringify(err.response.data));
			throw new Error(`Keycloak token request failed with status ${err.response.status}`);
		} else if (err.request) {
			console.error('Keycloak token fetch no response:', err.message);
			throw new Error('No response from Keycloak token endpoint');
		} else {
			console.error('Keycloak token fetch error:', err.message);
			throw new Error('Failed to request Keycloak token: ' + err.message);
		}
	}
}

async function getToken() {
	const now = Math.floor(Date.now() / 1000);
	// refresh 30s before expiry
	if (cached.access_token && cached.expires_at - 30 > now) {
		return cached.access_token;
	}
	await fetchToken();
	return cached.access_token;
}

async function getTokenInfo() {
	const now = Math.floor(Date.now() / 1000);
	return {
		hasToken: !!cached.access_token,
		expiresAt: cached.expires_at,
		secondsUntilExpiry: Math.max(0, cached.expires_at - now)
	};
}

module.exports = { getToken, getTokenInfo };
