use std::sync::Arc;

use quiver_db::Database;
use quiver_http::{HttpOptions, router, serve_tls, tls_config};
use rcgen::{CertificateParams, CertifiedIssuer, KeyPair};

struct Pki {
    ca: CertifiedIssuer<'static, KeyPair>,
    server_cert: String,
    server_key: String,
}

/// A CA and a server certificate for `localhost` that it signed.
fn pki() -> Pki {
    let mut ca_params = CertificateParams::new(Vec::new()).unwrap();
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().unwrap()).unwrap();
    let key = KeyPair::generate().unwrap();
    let cert = CertificateParams::new(vec!["localhost".into()])
        .unwrap()
        .signed_by(&key, &ca)
        .unwrap();
    Pki {
        server_cert: cert.pem(),
        server_key: key.serialize_pem(),
        ca,
    }
}

async fn start(pki: &Pki, client_ca: Option<&str>) -> u16 {
    let config = tls_config(
        pki.server_cert.as_bytes(),
        pki.server_key.as_bytes(),
        client_ca.map(str::as_bytes),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = router(Arc::new(Database::in_memory()), HttpOptions::default());
    tokio::spawn(serve_tls(listener, config, app, std::future::pending()));
    port
}

fn client(pki: &Pki) -> reqwest::ClientBuilder {
    let ca = reqwest::Certificate::from_pem(pki.ca.pem().as_bytes()).unwrap();
    reqwest::Client::builder()
        .tls_built_in_root_certs(false)
        .add_root_certificate(ca)
}

#[tokio::test]
async fn serves_https() {
    let pki = pki();
    let port = start(&pki, None).await;
    let url = format!("https://localhost:{port}/health");

    let res = client(&pki).build().unwrap().get(&url).send().await.unwrap();
    assert_eq!(res.status(), 200);

    // Plain HTTP and untrusted certificates are refused.
    assert!(
        reqwest::get(format!("http://localhost:{port}/health"))
            .await
            .is_err()
    );
    let untrusted = reqwest::Client::builder()
        .tls_built_in_root_certs(false)
        .build()
        .unwrap();
    assert!(untrusted.get(&url).send().await.is_err());
}

#[tokio::test]
async fn a_silent_client_does_not_block_others() {
    let pki = pki();
    let port = start(&pki, None).await;
    // Opens a connection but never starts the handshake.
    let _idle = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let res = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client(&pki)
            .build()
            .unwrap()
            .get(format!("https://localhost:{port}/health"))
            .send(),
    )
    .await
    .expect("served while another handshake is pending")
    .unwrap();
    assert_eq!(res.status(), 200);
}

#[tokio::test]
async fn mutual_tls_requires_a_client_certificate() {
    let pki = pki();
    let port = start(&pki, Some(&pki.ca.pem())).await;
    let url = format!("https://localhost:{port}/health");
    assert!(client(&pki).build().unwrap().get(&url).send().await.is_err());

    let key = KeyPair::generate().unwrap();
    let cert = CertificateParams::new(vec!["client".into()])
        .unwrap()
        .signed_by(&key, &pki.ca)
        .unwrap();
    let identity =
        reqwest::Identity::from_pem(format!("{}{}", cert.pem(), key.serialize_pem()).as_bytes()).unwrap();
    let res = client(&pki)
        .identity(identity)
        .build()
        .unwrap()
        .get(&url)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
}

#[test]
fn bad_pem_is_rejected() {
    let pki = pki();
    assert!(tls_config(b"nonsense", pki.server_key.as_bytes(), None).is_err());
    assert!(tls_config(pki.server_cert.as_bytes(), b"nonsense", None).is_err());
    assert!(
        tls_config(
            pki.server_cert.as_bytes(),
            pki.server_key.as_bytes(),
            Some(b"nonsense")
        )
        .is_err()
    );
    let other = KeyPair::generate().unwrap().serialize_pem();
    assert!(
        tls_config(pki.server_cert.as_bytes(), other.as_bytes(), None).is_err(),
        "mismatched key"
    );
}
