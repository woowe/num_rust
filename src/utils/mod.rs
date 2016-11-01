use Matrix2d;
use std::cmp;

use rayon;
use num_cpus;

pub fn sum_vec(vec: &[f64]) -> f64 {
    let mut mc = vec.clone();
    unrolled_sum(&mut mc)
}

pub fn frobenius_norm(m: &Matrix2d) -> f64 {
    let mut mc = vec_bin_op(m.get_matrix(), m.get_matrix(), |x, y| x * y);
    unrolled_sum(&mut mc).sqrt()
}

// from rulinalg, originally from bluss / ndarray
pub fn unrolled_sum(mut xs: &[f64]) -> f64
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut sum = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);
    while xs.len() >= 8 {
        p0 = p0 + xs[0].clone();
        p1 = p1 + xs[1].clone();
        p2 = p2 + xs[2].clone();
        p3 = p3 + xs[3].clone();
        p4 = p4 + xs[4].clone();
        p5 = p5 + xs[5].clone();
        p6 = p6 + xs[6].clone();
        p7 = p7 + xs[7].clone();

        xs = &xs[8..];
    }
    sum = sum.clone() + (p0 + p4);
    sum = sum.clone() + (p1 + p5);
    sum = sum.clone() + (p2 + p6);
    sum = sum.clone() + (p3 + p7);
    for elt in xs {
        sum = sum.clone() + elt.clone();
    }
    sum
}

// from rulinalg
pub fn vec_bin_op<F>(u: &[f64], v: &[f64], f: F) -> Vec<f64>
    where F: Fn(f64, f64) -> f64
{
    debug_assert_eq!(u.len(), v.len());
    let len = cmp::min(u.len(), v.len());

    let xs = &u[..len];
    let ys = &v[..len];

    let mut out_vec = Vec::with_capacity(len);
    unsafe {
        out_vec.set_len(len);
    }

    {
        let out_slice = &mut out_vec[..len];

        for i in 0..len {
            out_slice[i] = f(xs[i], ys[i]);
        }
    }

    out_vec
}

pub fn vec_bin_op_mut<F>(u: &[f32], v: &[f32], len: usize, dst: &mut [f32], f: &F) -> ()
where F: Fn(f32, f32) -> f32 + Send + Sync + 'static
{
    let mut x_iter = u.iter();
    let mut y_iter = v.iter();

    for dst in dst.iter_mut() {
        *dst = f(*x_iter.next().unwrap(), *y_iter.next().unwrap());
    }
}

pub fn get_chunk_size<T>(u: &[T], v: &[T]) -> usize {
    debug_assert_eq!(u.len(), v.len());
    let len = cmp::min(u.len(), v.len());
    let cpus = num_cpus::get();
    let mut chunk_size = (len as f32 / cpus as f32).floor() as usize;
    if len < cpus {
        chunk_size = len;
    }
    return chunk_size;
}


pub fn vec_bin_op_split<F>(u: &[f32], v: &[f32], dst: &mut [f32], chunk_size: &usize, f: &F) -> ()
    where F: Fn(f32, f32) -> f32 + Send + Sync + 'static
{
    // debug_assert!(u.len() == v.len());
    let len = u.len();

    if len < *chunk_size {
        // println!("LEN: {}", u.len());
        vec_bin_op_mut(u, v, len, dst, f);
        return;
    }

    let mid_point = len / 2;
    let (x_left, x_right): (&[f32], &[f32]) = u.split_at(mid_point);
    let (y_left, y_right): (&[f32], &[f32]) = v.split_at(mid_point);
    let (dst_left, dst_right): (&mut [f32], &mut [f32]) = dst.split_at_mut(mid_point);

    rayon::join(|| vec_bin_op_split(x_left, y_left, dst_left, chunk_size, f),
             || vec_bin_op_split(x_right, y_right, dst_right, chunk_size, f));
}

pub fn vec_bin_op_threaded<F>(u: &[f32], v: &[f32], chunk_size: &usize, f: &F) -> Vec<f32>
    where F: Fn(f32, f32) -> f32 + Send + Sync + 'static
{
    let len = u.len();
    debug_assert!(len == v.len());

    let mut out_vec = Vec::with_capacity(len);
    unsafe {
        out_vec.set_len(len);
    }

    vec_bin_op_split(u, v, &mut out_vec, chunk_size, f);

    return out_vec;
}


pub fn vec_fn_op_mut<F>(u: &[f64], dst: &mut [f64], f: &F) -> ()
where F: Fn(f64) -> f64 + Send + Sync + 'static
{
    let mut x_iter = u.iter();

    for dst in dst.iter_mut() {
        *dst = f(*x_iter.next().unwrap());
    }
}


pub fn vec_fn_op_split<F>(u: &[f64], dst: &mut [f64], chunk_size: &usize, f: &F) -> ()
    where F: Fn(f64) -> f64 + Send + Sync + 'static
{
    // debug_assert!(u.len() == v.len());
    let len = u.len();

    if len < *chunk_size {
        // println!("LEN: {}", u.len());
        vec_fn_op_mut(u, dst, f);
        return;
    }

    let mid_point = len / 2;
    let (x_left, x_right): (&[f64], &[f64]) = u.split_at(mid_point);
    let (dst_left, dst_right): (&mut [f64], &mut [f64]) = dst.split_at_mut(mid_point);

    rayon::join(|| vec_fn_op_split(x_left, dst_left, chunk_size, f),
             || vec_fn_op_split(x_right, dst_right, chunk_size, f));
}

pub fn vec_fn_op_threaded<F>(u: &[f64], chunk_size: &usize, f: &F) -> Vec<f64>
    where F: Fn(f64) -> f64 + Send + Sync + 'static
{
    let len = u.len();

    let mut out_vec = Vec::with_capacity(len);
    unsafe {
        out_vec.set_len(len);
    }

    vec_fn_op_split(u, &mut out_vec, chunk_size, f);

    out_vec
}
