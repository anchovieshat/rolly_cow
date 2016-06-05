#[macro_use]
extern crate glium;
extern crate image;

use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use std::io::Cursor;
use std::f32;

use glium::{DisplayBuild, Surface};

#[derive(Copy, Clone, Debug)]
struct Quaternion {
    rotation: [f32; 4],
}

impl Quaternion {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quaternion {
        Quaternion {
            rotation: [x, y, z, w],
        }
    }

    pub fn from_angle(angle: f32, x: f32, y: f32, z: f32) -> Quaternion {
        Quaternion {
            rotation: [(angle / 2.0).cos(), (angle / 2.0).sin() * (x).cos(), (angle / 2.0).sin() * (y).cos(), (angle / 2.0).sin() * (z).cos()],
        }
    }

    pub fn from_mat4(matrix: [[f32; 4]; 4]) -> Quaternion {
        let mut w = (1.0 + matrix[0][0] + matrix[1][1] + matrix[2][2]).sqrt() / 2.0;
        let mut x = (matrix[2][1] - matrix[1][2]) / (4.0 * w);
        let mut y = (matrix[0][2] - matrix[2][0]) / (4.0 * w);
        let mut z = (matrix[1][0] - matrix[0][1]) / (4.0 * w);

        let rot_sq = (x * x) + (y * y) + (z * z) + (w * w);
        if rot_sq > 0.1 {
            let inv = 1.0 / rot_sq.sqrt();
            x = x * inv;
            y = y * inv;
            z = z * inv;
            w  = w * inv;
        }
        Quaternion {
            rotation: [x, y, z, w],
        }
    }

    pub fn normalize(&mut self) {
        let x = 0;
        let y = 1;
        let z = 2;
        let w = 3;

        let rot_sq = (self.rotation[x] * self.rotation[x]) + (self.rotation[y] * self.rotation[y]) + (self.rotation[z] * self.rotation[z]) + (self.rotation[w] * self.rotation[w]);
        if rot_sq > 0.1 {
            let inv = 1.0 / rot_sq.sqrt();
            self.rotation = [self.rotation[x] * inv, self.rotation[y] * inv, self.rotation[z] * inv, self.rotation[w] * inv];
        }
    }

    pub fn to_mat4(&self) -> [[f32; 4]; 4] {
        let x = 0;
        let y = 1;
        let z = 2;
        let w = 3;

        let m00 = 1.0 - (2.0 * self.rotation[z] * self.rotation[z]) - (2.0 * self.rotation[y] * self.rotation[y]);
        let m01 = (-2.0 * self.rotation[z] * self.rotation[w]) + (2.0 * self.rotation[y] * self.rotation[x]);
        let m02 = (2.0 * self.rotation[y] * self.rotation[w]) - (2.0 * self.rotation[z] * self.rotation[x]);

        let m10 = (2.0 * self.rotation[x] * self.rotation[y]) + (2.0 * self.rotation[w] * self.rotation[z]);
        let m11 = 1.0 - (2.0 * self.rotation[z] * self.rotation[z]) + (2.0 * self.rotation[x] * self.rotation[x]);
        let m12 = (2.0 * self.rotation[z] * self.rotation[y]) - (2.0 * self.rotation[x] * self.rotation[w]);

        let m20 = (2.0 * self.rotation[x] * self.rotation[z]) - (2.0 * self.rotation[w] * self.rotation[z]);
        let m21 = (2.0 * self.rotation[y] * self.rotation[z]) + (2.0 * self.rotation[w] * self.rotation[x]);
        let m22 = 1.0 - (2.0 * self.rotation[y] * self.rotation[y]) - (2.0 * self.rotation[z] * self.rotation[x]);

        [
            [m00, m01, m02, 0.0],
            [m10, m11, m12, 0.0],
            [m20, m21, m22, 0.0],
            [0.0, 0.0, 0.0, 1.0f32],
        ]
    }
}

pub fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn rotate_x(angle: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, angle.cos(), angle.sin(), 0.0],
        [0.0, -angle.sin(), angle.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn rotate_y(angle: f32) -> [[f32; 4]; 4] {
    [
        [angle.cos(), 0.0, -angle.sin(), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [angle.sin(), 0.0, angle.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn rotate_z(angle: f32) -> [[f32; 4]; 4] {
    [
        [angle.cos(), angle.sin(), 0.0, 0.0],
        [-angle.sin(), angle.cos(), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn simd_pack(mat: [[f32; 4]; 4]) -> [f32x4; 4] {
    let r1 = f32x4::new(mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
    let r2 = f32x4::new(mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
    let r3 = f32x4::new(mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
    let r4 = f32x4::new(mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
    [r1, r2, r3, r4]
}

pub fn simd_unpack(mat: [f32x4; 4]) -> [[f32; 4]; 4] {
    [
        [mat[0].extract(0), mat[0].extract(1), mat[0].extract(2), mat[0].extract(3)],
        [mat[1].extract(0), mat[1].extract(1), mat[1].extract(2), mat[1].extract(3)],
        [mat[2].extract(0), mat[2].extract(1), mat[2].extract(2), mat[2].extract(3)],
        [mat[3].extract(0), mat[3].extract(1), mat[3].extract(2), mat[3].extract(3)],
    ]
}

pub fn simd_mul_mat4(m1: [f32x4; 4], m2: [f32x4; 4]) -> [f32; 4] {
    [m1[0] * m2[0], m1[1] * m2[1], m1[2] * m2[2], m1[3] * m2[3]]
}

pub fn mul_mat4(m1: [[f32; 4]; 4], m2: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m1[0][0] * m2[0][0], m1[0][1] * m2[0][1], m1[0][2] * m2[0][2], m1[0][3] * m2[0][3]],
        [m1[1][0] * m2[1][0], m1[1][1] * m2[1][1], m1[1][2] * m2[1][2], m1[1][3] * m2[1][3]],
        [m1[2][0] * m2[2][0], m1[2][1] * m2[2][1], m1[2][2] * m2[2][2], m1[2][3] * m2[2][3]],
        [m1[3][0] * m2[3][0], m1[3][1] * m2[3][1], m1[3][2] * m2[3][2], m1[3][3] * m2[3][3]],
    ]
}

#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
}

implement_vertex!(Vertex, position);

#[derive(Copy, Clone, Debug)]
pub struct Normal {
    normal: [f32; 3],
}

implement_vertex!(Normal, normal);

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
	let f = {
		let f = direction;
		let len = ((f[0] * f[0]) + (f[1] * f[1]) + (f[2] * f[2])).sqrt();
		[f[0] / len, f[1] / len, f[2] / len]
	};

	let s = [(up[1] * f[2]) - (up[2] * f[1]),
			 (up[2] * f[0]) - (up[0] * f[2]),
			 (up[0] * f[1]) - (up[1] * f[0])];

	let s_norm = {
		let len = ((s[0] * s[0]) + (s[1] * s[1]) + (s[2] * s[2])).sqrt();
		[s[0] / len, s[1] / len, s[2] / len]
	};

	let u = [(f[1] * s_norm[2]) - (f[2] * s_norm[1]),
			 (f[2] * s_norm[0]) - (f[0] * s_norm[2]),
			 (f[0] * s_norm[1]) - (f[1] * s_norm[0])];

	let p = [(-position[0] * s_norm[0]) - (position[1] * s_norm[1]) - (position[2] * s_norm[2]),
			 (-position[0] * u[0]) - (position[1] * u[1]) - (position[2] * u[2]),
			 (-position[0] * f[0]) - (position[1] * f[1]) - (position[2] * f[2])];

	[
		[s[0], u[0], f[0], 0.0],
		[s[1], u[1], f[1], 0.0],
		[s[2], u[2], f[2], 0.0],
		[p[0], p[1], p[2], 1.0],
	]
}



fn main() {
    let mut width = 640;
    let mut height = 480;
    let mut ratio = height as f32 / width as f32;
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title(format!("Rolly Cow"))
        .with_depth_buffer(32)
        .with_vsync()
        .build_glium().unwrap();

    let input = BufReader::new(File::open("assets/cow.obj").unwrap());
    let mut verts_packed = Vec::new();
    let mut normals_packed = Vec::new();
    let mut verts: Vec<Vertex> = Vec::new();
    let mut normals: Vec<Normal> = Vec::new();
    let mut faces = Vec::new();
    let mut idx: u32 = 0;
    for line in input.lines() {
        let line = line.unwrap();
        if line.contains("v ") {
            let pieces: Vec<&str> = line.split_whitespace().collect();
            let p1: f32 = pieces[1].parse().unwrap();
            let p2: f32 = pieces[2].parse().unwrap();
            let p3: f32 = pieces[3].parse().unwrap();
            let vert = Vertex { position: [p1, p2, p3] };
            verts_packed.push(vert);
        } else if line.contains("vn ") {
            let pieces: Vec<&str> = line.split_whitespace().collect();
            let p1: f32 = pieces[1].parse().unwrap();
            let p2: f32 = pieces[2].parse().unwrap();
            let p3: f32 = pieces[3].parse().unwrap();
            let norm = Normal { normal: [p1, p2, p3] };
            normals_packed.push(norm);
        } else if line.contains("f ") {
            let indices: Vec<&str> = line.split_whitespace().collect();
            for index in indices {
                if !index.contains("f") {
                    let i: Vec<&str> = index.split("//").collect();
                    let vert_idx: usize = i[0].parse().unwrap();
                    let norm_idx: usize = i[1].parse().unwrap();
                    let vert_idx = vert_idx - 1;
                    let norm_idx = norm_idx - 1;

                    normals.push(normals_packed[norm_idx]);
                    verts.push(verts_packed[vert_idx]);

                    faces.push(idx);
                    idx += 1;
                }
            }
        }
    }

    let image = image::load(Cursor::new(&include_bytes!("../assets/cow.png")[..]), image::PNG).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(image.into_raw(), image_dimensions);
    let texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();

    let vertex_buffer = glium::VertexBuffer::new(&display, &verts).unwrap();
    let normal_buffer = glium::VertexBuffer::new(&display, &normals).unwrap();

    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &faces).unwrap();

    let vertex_shader_src = r#"
        #version 150

        in vec3 position;
        in vec3 normal;

        out vec3 v_normal;
        out vec3 v_position;

        uniform mat4 rotation;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            mat4 rotmod = model * rotation;
            v_normal = transpose(inverse(mat3(rotmod))) * normal;
            gl_Position = perspective * view * rotmod * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;

    let fragment_shader_src = r#"
        #version 150

        in vec3 v_normal;
        in vec3 v_position;

        out vec4 color;
        uniform vec3 u_light;

        const vec3 ambient_color = vec3(0.2, 0.0, 0.0);
        const vec3 diffuse_color = vec3(0.6, 0.2, 0.0);
        const vec3 specular_color = vec3(1.0, 1.0, 1.0);

        const float A = 0.1;
        const float B = 0.3;
        const float C = 0.6;
        const float D = 1.0;

        float stepmix(float edge0, float edge1, float E, float x) {
            float T = clamp(0.5 * (x - edge0 + E) / E, 0.0, 1.0);
            return mix(edge0, edge1, T);
        }

        void main() {
            float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);
            float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);

            float E = fwidth(diffuse);
            if (diffuse > A - E && diffuse < A + E) {
                diffuse = stepmix(A, B, E, diffuse);
            } else if (diffuse > B - E && diffuse < B + E) {
                diffuse = stepmix(B, C, E, diffuse);
            } else if (diffuse > C - E && diffuse < C + E) {
                diffuse = stepmix(C, D, E, diffuse);
            } else if (diffuse < A) {
                diffuse = 0.0;
            } else if (diffuse < B) {
                diffuse = B;
            } else if (diffuse < C) {
                diffuse = C;
            } else {
                diffuse = D;
            }

            E = fwidth(specular);
            if (specular > 0.5 - E && specular < 0.5 + E) {
                specular = smoothstep(0.5 - E, 0.5 + E, specular);
            } else {
                specular = step(0.5, specular);
            }

            color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let two_pi = 2.0 * f32::consts::PI;
    let mut rot: f32 = 0.0;
    let mut t = -5.0;
    loop {
        for event in display.poll_events() {
            match event {
                glium::glutin::Event::Closed => return,
                glium::glutin::Event::KeyboardInput(state, _, key) => {
                    if key.is_some() {
                        let key = key.unwrap();
                        println!("{:?}", key);
                    }
                },
                glium::glutin::Event::Resized(tmp_height, tmp_width) => {
                    println!("{},{}", tmp_height, tmp_width);
					height = tmp_height;
					width = tmp_width;
					ratio = (height as f32) / (width as f32);
				},
                _ => (),
            }
        }

        t += 0.002;
        rot -= 0.002;
        if t > 5.0 {
            t = -5.0;
        }

        let light = [1.0, 1.0, -1.0f32];
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        let perspective = {
            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;
            let f = 1.0 / (fov / 2.0).tan();

            [
                [f * ratio, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (zfar+znear)/(zfar-znear), 1.0],
                [0.0, 0.0, -(2.0*zfar*znear)/(zfar-znear), 0.0],
            ]
        };

        let view = view_matrix(&[0.0, 0.0, -3.0], &[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0]);

        let uniforms = uniform! {
            rotation: mul_mat4(rotate_y(t), rotate_z(t)),
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32],
            ],
            view: view,
            perspective: perspective,
            u_light: light,
        };

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            dithering: false,
            smooth: Some(glium::draw_parameters::Smooth::Nicest),
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
            .. Default::default()
        };

        target.draw((&vertex_buffer, &normal_buffer), &indices, &program, &uniforms, &params).unwrap();

        target.finish().unwrap();
    }
}
