use std::{
    env,
    fmt::Write,
    io::Cursor,
    thread,
    time::{Duration, Instant},
};

use async_compat::Compat;
use image::io::Reader as ImageReader;
use inline_python::{python, Context};
use serde::Deserialize;
use smol::prelude::*;
use telegram_bot::prelude::*;
use telegram_bot::{Api, MessageKind, UpdateKind};

#[macro_use]
extern crate log;

fn main() {
    log4rs::init_file("log_config.yml", Default::default()).expect("No se pudo iniciar Log");
    info!("Iniciando...");

    let token = env::var("POKE_TOKEN").expect("Token no encontrado");
    let api = Api::new(&token);
    let mut stream = api.stream();

    let client = reqwest::Client::new();

    let mut telegram_url = String::from("https://api.telegram.org/");
    let original_len = telegram_url.len();

    let c: Context = python! {
        from fastbook import load_learner
        from pathlib import Path

        learn_inf = load_learner(Path()/"export.pkl")
    };

    info!("Listo para recibir querys.");

    smol::block_on(Compat::new(async {
        while let Some(update) = stream.next().await {
            match update {
                Ok(update) => {
                    if let UpdateKind::Message(mut message) = update.kind {
                        let now = Instant::now();

                        if let MessageKind::Photo { ref mut data, .. } = message.kind {
                            write!(
                                telegram_url,
                                "bot{}/getFile?file_id={}",
                                &token, &data[0].file_id
                            )
                            .unwrap();

                            let response = client
                                .get(&telegram_url)
                                .send()
                                .await
                                .unwrap()
                                .json::<Response>()
                                .await
                                .unwrap();

                            telegram_url.truncate(original_len);

                            write!(
                                telegram_url,
                                "file/bot{}/{}",
                                &token, &response.result.file_path
                            )
                            .unwrap();

                            let image_bytes = client
                                .get(&telegram_url)
                                .send()
                                .await
                                .unwrap()
                                .bytes()
                                .await
                                .unwrap();

                            let img = ImageReader::new(Cursor::new(image_bytes))
                                .with_guessed_format()
                                .unwrap()
                                .decode()
                                .unwrap();

                            img.resize_exact(512, 512, image::imageops::FilterType::Nearest);

                            img.save("test.jpg").unwrap();

                            c.run(python!{
                                prediction = learn_inf.predict("test.jpg")
                                ans = f"{prediction[0]} with a confidence of {max(prediction[-1]):.4}".title()
                            });

                            api.spawn(message.text_reply(c.get::<String>("ans")));

                            info!(
                                "{} {:#?}",
                                &message.from.first_name,
                                Instant::now().duration_since(now)
                            );

                            telegram_url.truncate(original_len);
                        }
                    }
                }

                Err(e) => {
                    error!("{}", e);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        }
    }));
}

#[derive(Deserialize)]
struct Response {
    result: Result,
}

#[derive(Deserialize)]
struct Result {
    file_path: String,
}
