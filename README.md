# OpenAlexChromeExtension
Chrome extension + FastAPI backend that discovers ML engineering candidates from academic literature. Seeds a citation graph via OpenAlex, scores authors on systems/infra signal using POS/NEG keyword matching across 6 buckets, applies an engineering gate, and optionally runs a GitHub industry-signal sniff test. Exports ranked candidates as XLSX.
