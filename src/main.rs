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
use telegram_bot::{
    prelude::*, reply_markup, Api, CanSendMessage, MessageKind, ParseMode, UpdateKind,
};

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
    let mut ans: Vec<String> = Vec::new();
    let mut text = String::new();

    let telegram_len = telegram_url.len();

    let c: Context = python! {
        from fastbook import load_learner

        learn_inf = load_learner("./export.pkl")
    };

    info!("Listo para recibir querys.");

    smol::block_on(Compat::new(async {
        while let Some(update) = stream.next().await {
            match update {
                Ok(update) => {
                    if let UpdateKind::Message(mut message) = update.kind {
                        let now = Instant::now();
                        match message.kind {
                            MessageKind::Photo { ref mut data, .. } => {
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

                                telegram_url.truncate(telegram_len);

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

                                let img = ImageReader::with_format(
                                    Cursor::new(image_bytes),
                                    image::ImageFormat::Jpeg,
                                )
                                .decode()
                                .unwrap();

                                img.resize_exact(680, 680, image::imageops::FilterType::Nearest);

                                img.save_with_format("i", image::ImageFormat::Jpeg).unwrap();

                                c.run(python! {
                                    prediction = learn_inf.predict("i")
                                    ans = f"{prediction[0]}", f"{max(prediction[-1]):.4}"
                                });

                                ans = c.get::<Vec<String>>("ans");

                                let pokelink =
                                    format!("https://www.pokemon.com/us/pokedex/{}", ans[0]);

                                write!(
                                    &mut text,
                                    "[{}]({}), confidence {}",
                                    ans[0], pokelink, ans[1],
                                )
                                .unwrap();

                                text[..=1].make_ascii_uppercase();

                                let mut reply = message.text_reply(&text);
                                reply.parse_mode(ParseMode::Markdown);
                                reply.reply_markup(
                                    reply_markup!(inline_keyboard, ["Pokédex" url pokelink]),
                                );

                                api.spawn(reply);

                                telegram_url.truncate(telegram_len);
                                text.clear();
                            }

                            MessageKind::Text { ref mut data, .. } => {
                                if data.contains("/h") {
                                    api.spawn(message.chat.text(HELP));
                                } else if data.contains("/s") {
                                    api.spawn(
                                        message.chat.text(START).parse_mode(ParseMode::Markdown),
                                    );
                                } else {
                                    api.spawn(message.chat.text(NO_MATCH));
                                }
                            }

                            _ => {
                                api.spawn(message.chat.text(NO_MATCH));
                            }
                        }

                        info!(
                            "{} {} {} {:#?}",
                            &message.from.first_name,
                            ans[0],
                            ans[1],
                            Instant::now().duration_since(now)
                        );
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

static START: &str =
    "*PokeClass*\n\nJust send a picture of a Pokémon and I will try to guess which one it is using deep learning";
static HELP: &str = "Just send a picture";
static NO_MATCH: &str = "I didn't understand your message, try again or use /help";
