#![allow(dead_code)]

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{
        canvas::{Canvas, Points},
        Widget,
    },
};

use super::theme::Palette;
use super::worldmap;

/// A spinning ASCII globe rendered using braille unicode characters.
pub struct Globe {
    /// Current rotation angle around the Y axis (radians).
    angle: f64,
    /// Rotation speed in radians per tick.
    rotation_speed: f64,
    /// 3D cartesian coordinates of continent points on the unit sphere.
    continent_points: Vec<(f64, f64, f64)>,
    /// 3D coordinates of active network nodes.
    node_positions: Vec<(f64, f64, f64)>,
    /// This node's position (if known).
    my_position: Option<(f64, f64, f64)>,
    /// Throughput sparkline history for subtle pulsing.
    pulse_phase: f64,
}

impl Globe {
    pub fn new() -> Self {
        let raw_points = worldmap::get_world_points();
        let continent_points: Vec<(f64, f64, f64)> = raw_points
            .iter()
            .map(|&(lat, lon)| latlon_to_xyz(lat, lon))
            .collect();

        Self {
            angle: 0.0,
            rotation_speed: 0.02, // ~1 revolution per 314 ticks (~31s at 10fps)
            continent_points,
            node_positions: Vec::new(),
            my_position: None,
            pulse_phase: 0.0,
        }
    }

    /// Advance the globe rotation by one tick.
    pub fn tick(&mut self) {
        self.tick_active(false);
    }

    /// Advance the globe rotation, accelerating while a request is active.
    pub fn tick_active(&mut self, active: bool) {
        let speed = if active {
            self.rotation_speed * 3.0
        } else {
            self.rotation_speed
        };
        self.angle += speed;
        if self.angle > std::f64::consts::TAU {
            self.angle -= std::f64::consts::TAU;
        }
        self.pulse_phase += if active { 0.18 } else { 0.1 };
    }

    /// Set the rotation angle directly (for startup animation).
    #[allow(dead_code)]
    pub fn set_angle(&mut self, angle: f64) {
        self.angle = angle;
    }

    /// Set mock node positions for demo purposes.
    pub fn set_mock_nodes(&mut self) {
        let node_coords = [
            (37.7749, -122.4194), // San Francisco
            (40.7128, -74.0060),  // New York
            (51.5074, -0.1278),   // London
            (48.8566, 2.3522),    // Paris
            (35.6762, 139.6503),  // Tokyo
            (1.3521, 103.8198),   // Singapore
            (-33.8688, 151.2093), // Sydney
            (55.7558, 37.6173),   // Moscow
            (19.4326, -99.1332),  // Mexico City
            (-23.5505, -46.6333), // Sao Paulo
            (25.2048, 55.2708),   // Dubai
            (37.5665, 126.9780),  // Seoul
            (52.5200, 13.4050),   // Berlin
            (43.6532, -79.3832),  // Toronto
            (-1.2921, 36.8219),   // Nairobi
        ];

        self.node_positions = node_coords
            .iter()
            .map(|&(lat, lon)| latlon_to_xyz(lat, lon))
            .collect();

        // User's node (San Francisco for demo)
        self.my_position = Some(latlon_to_xyz(37.7749, -122.4194));
    }

    /// Set node positions from region strings (from orchestrator discovery).
    /// Each region is mapped to approximate coordinates. Nodes with unknown
    /// regions are spread across the globe using a hash of their wallet address.
    pub fn set_nodes_from_regions(&mut self, regions: &[(String, Option<String>)]) {
        self.node_positions = regions
            .iter()
            .map(|(wallet, region)| {
                let (lat, lon) = region_to_latlon(region.as_deref(), wallet);
                latlon_to_xyz(lat, lon)
            })
            .collect();
    }

    /// Set this node's position from its region.
    pub fn set_my_position(&mut self, region: Option<&str>, wallet: &str) {
        let (lat, lon) = region_to_latlon(region, wallet);
        self.my_position = Some(latlon_to_xyz(lat, lon));
    }

    /// Project a 3D point with the current rotation, returning 2D coords if visible.
    fn project(&self, point: (f64, f64, f64)) -> Option<(f64, f64)> {
        let (x, y, z) = point;
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        // Y-axis rotation
        let rx = x * cos_a - z * sin_a;
        let rz = x * sin_a + z * cos_a;
        let ry = y;

        // Only render front-facing points
        if rz > -0.1 {
            // Slight depth perspective
            let scale = 1.0 + rz * 0.15;
            Some((rx * scale, ry * scale))
        } else {
            None
        }
    }

    /// Render the globe into a ratatui Canvas widget.
    pub fn render(&self, area: Rect, buf: &mut Buffer, palette: Palette) {
        if area.width < 4 || area.height < 4 {
            return;
        }

        let visible_continents: Vec<(f64, f64)> = self
            .continent_points
            .iter()
            .filter_map(|&p| self.project(p))
            .collect();

        let visible_nodes: Vec<(f64, f64)> = self
            .node_positions
            .iter()
            .filter_map(|&p| self.project(p))
            .collect();

        let visible_me: Vec<(f64, f64)> = self
            .my_position
            .iter()
            .filter_map(|&p| self.project(p))
            .collect();

        // To make the globe appear circular, we set the canvas bounds so that
        // one unit in x covers the same physical distance as one unit in y.
        //
        // Braille resolution: each char = 2 dots wide, 4 dots tall.
        // Physical char cell is ~2x taller than wide (aspect ~0.5 w:h).
        // So each braille dot: width = cell_w/2, height = cell_h/4 = (2*cell_w)/4 = cell_w/2.
        // Braille dots are roughly square — but line spacing makes rows taller.
        //
        // We measure the physical aspect of the render area and set bounds accordingly.
        // char_ratio = physical width of a char / physical height of a char ≈ 0.5
        let char_ratio = 0.51;
        let physical_w = area.width as f64 * char_ratio;
        let physical_h = area.height as f64; // in char-height units

        let pad = 1.15;
        let (x_extent, y_extent) = if physical_w > physical_h {
            // Wider than tall: x range is larger
            (pad * physical_w / physical_h, pad)
        } else {
            // Taller than wide: y range is larger
            (pad, pad * physical_h / physical_w)
        };

        // Generate sphere outline (unit circle — canvas bounds handle aspect)
        let outline: Vec<(f64, f64)> = (0..120)
            .map(|i| {
                let theta = (i as f64 / 120.0) * std::f64::consts::TAU;
                (theta.cos(), theta.sin())
            })
            .collect();

        let canvas = Canvas::default()
            .x_bounds([-x_extent, x_extent])
            .y_bounds([-y_extent, y_extent])
            .paint(move |ctx| {
                ctx.draw(&Points {
                    coords: &outline,
                    color: palette.globe_outline,
                });
                ctx.draw(&Points {
                    coords: &visible_continents,
                    color: palette.globe_land,
                });
                ctx.draw(&Points {
                    coords: &visible_nodes,
                    color: palette.globe_nodes,
                });
                ctx.draw(&Points {
                    coords: &visible_me,
                    color: palette.globe_me,
                });
            })
            .marker(ratatui::symbols::Marker::Braille);

        canvas.render(area, buf);
    }
}

/// Map a region string (or cloud region code) to approximate lat/lon.
/// Falls back to a deterministic pseudo-random position based on wallet address hash.
fn region_to_latlon(region: Option<&str>, wallet: &str) -> (f64, f64) {
    if let Some(r) = region {
        let r = r.to_lowercase();
        // Match common cloud region codes and geographic names
        if r.contains("us-east") || r.contains("virginia") || r.contains("new-york") {
            return (39.0, -77.5);
        }
        if r.contains("us-west") || r.contains("oregon") || r.contains("california") {
            return (37.8, -122.4);
        }
        if r.contains("us-central") || r.contains("iowa") {
            return (41.9, -93.6);
        }
        if r.contains("eu-west") || r.contains("ireland") || r.contains("london") {
            return (51.5, -0.1);
        }
        if r.contains("eu-central") || r.contains("frankfurt") || r.contains("germany") {
            return (50.1, 8.7);
        }
        if r.contains("ap-southeast") || r.contains("singapore") {
            return (1.4, 103.8);
        }
        if r.contains("ap-northeast") || r.contains("tokyo") || r.contains("japan") {
            return (35.7, 139.7);
        }
        if r.contains("ap-south") || r.contains("mumbai") || r.contains("india") {
            return (19.1, 72.9);
        }
        if r.contains("sa-east") || r.contains("sao-paulo") || r.contains("brazil") {
            return (-23.6, -46.6);
        }
        if r.contains("sydney") || r.contains("australia") {
            return (-33.9, 151.2);
        }
        if r.contains("seoul") || r.contains("korea") {
            return (37.6, 127.0);
        }
        if r.contains("canada") || r.contains("toronto") {
            return (43.7, -79.4);
        }
        if r.contains("africa") || r.contains("cape-town") {
            return (-33.9, 18.4);
        }
    }

    // Deterministic pseudo-random position from wallet address
    let hash: u64 = wallet
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let lat = ((hash % 1400) as f64 / 10.0) - 70.0; // -70 to +70
    let lon = (((hash / 1400) % 3600) as f64 / 10.0) - 180.0; // -180 to +180
    (lat, lon)
}

/// Convert latitude/longitude (degrees) to 3D cartesian coordinates on a unit sphere.
fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let x = lat.cos() * lon.cos();
    let y = lat.sin();
    let z = lat.cos() * lon.sin();
    (x, y, z)
}
