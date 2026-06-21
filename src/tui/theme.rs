#![allow(dead_code)]

use ratatui::style::Color;

#[derive(Clone, Copy)]
pub struct Palette {
    pub text: Color,
    pub muted: Color,
    pub dim: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
    pub globe_outline: Color,
    pub globe_land: Color,
    pub globe_nodes: Color,
    pub globe_me: Color,
}

pub fn palette() -> Palette {
    Palette {
        text: Color::White,
        muted: Color::Gray,
        dim: Color::DarkGray,
        success: Color::Green,
        warning: Color::Yellow,
        danger: Color::Red,
        globe_outline: Color::DarkGray,
        globe_land: Color::White,
        globe_nodes: Color::Green,
        globe_me: Color::Yellow,
    }
}
