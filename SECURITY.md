# Security Policy

## Reporting a vulnerability

Do not publish API keys, credentials, private documents, or exploit details in a public issue. For a personal repository, contact the repository owner privately using the contact method listed on their GitHub profile.

## Deployment boundary

The current application is designed for local development. It does not implement user authentication or authorization, and the embedded Streamlit configuration disables CORS and XSRF protections. Do not expose it to untrusted networks until those risks have been addressed.
