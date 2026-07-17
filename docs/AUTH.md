# Authentication

The backend supports two authentication modes selected via `AUTH_MODE`.

## API key mode (default)

`AUTH_MODE=api_key`

Protected endpoints accept:

- `X-API-Key` header matching `API_KEY`, or
- HttpOnly session cookie from `POST /auth/session`, or
- (optional) `Authorization: Bearer <jwt>` for transitional use

In `development`/`local` mode, endpoints are open when `API_KEY` is unset (warning logged).

## OAuth2/JWT mode

`AUTH_MODE=oauth2`

Protected endpoints prefer `Authorization: Bearer <jwt>`. API key and session cookie remain supported for backward compatibility during migration.

### Demo token endpoint

`POST /auth/token` with JSON body:

```json
{"username": "analyst@example.com", "password": "<API_KEY>"}
```

Returns:

```json
{"access_token": "...", "token_type": "bearer", "expires_in": 3600}
```

This is a **demo password-grant stub**, not a full OAuth2 authorization server. Production multi-tenant deployments should integrate with an identity provider (Auth0, Azure AD, Keycloak) and validate JWTs from that issuer.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTH_MODE` | `api_key` | `api_key` or `oauth2` |
| `JWT_SECRET` | falls back to `SESSION_SECRET` | HMAC signing key |
| `JWT_TTL_SECONDS` | `3600` | Access token lifetime |
| `JWT_ISSUER` | `credit-risk-pd-engine` | Token `iss` claim |
| `JWT_AUDIENCE` | `credit-risk-api` | Token `aud` claim |

### Audit attribution

When authenticated via JWT, the token `sub` claim is stored in `predictions.user_id` in the audit store.

## Migration guidance

1. Deploy with `AUTH_MODE=api_key` (current behavior).
2. Configure `JWT_SECRET` and test `POST /auth/token` in a staging environment.
3. Update clients to send Bearer tokens.
4. Switch to `AUTH_MODE=oauth2` when all clients support JWT.
5. For enterprise SSO, replace `/auth/token` validation with your IdP JWKS endpoint.

See [SCALING.md](SCALING.md) for multi-instance deployment notes.
