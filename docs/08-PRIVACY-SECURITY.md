# 08 — Privacy & Security

## 1. Threat model

Assume an attacker may attempt to obtain:

- raw conversations;
- memories;
- embeddings;
- exported data;
- database backups;
- API keys;
- model prompts;
- tool credentials.

## 2. Privacy modes

### Strict Local

```text
LLM              local
embeddings       local
database         local
memory           local
tools            local where possible
cloud inference  blocked
```

### Hybrid

```text
Sensitive data       local
Personal memory      local
General tasks        optional cloud
Public information   optional cloud
```

The router must enforce this policy before context is constructed.

## 3. Important security rule

Do not rely on the prompt saying:

> "Do not send private information."

The architecture must enforce it.

## 4. Encryption

At rest:

- encrypted database/storage where practical;
- OS-level disk encryption;
- secret/key separation.

In transit:

- HTTPS/TLS for network services;
- localhost-only binding for local model services where appropriate.

## 5. Logging

Default:

```text
log:
request_id
latency
model
token counts
tool name
error type
```

Do not default to:

```text
full user prompt
full retrieved memories
full model output
```

## 6. Privacy tests

Automated tests should verify:

- strict mode never calls cloud provider;
- sensitive memory never enters cloud request;
- deleted memories cannot be retrieved;
- unauthorized user cannot access another user's memory;
- logs do not contain raw sensitive text;
- exports contain only user-authorized data.

## 7. Data ownership

The product should make these actions easy:

- export all;
- inspect;
- edit;
- forget one memory;
- forget a category;
- delete everything.

## 8. Privacy claim

We should never claim:

> "100% private."

Instead:

> "In strict-local mode, personal data is processed and stored locally by the application, subject to the security of the user's device and operating environment."
