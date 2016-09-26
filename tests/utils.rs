extern crate num_rust;

use num_rust::ext::traits::ToMatrix2d;
use num_rust::utils::*;

#[test]
fn sum_vec_test() {
    assert!(10.0 == sum_vec(&vec![1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn vec_bin_op_test() {
    let m = vec![1., 2., 3., 4.];
    assert!(vec![1., 4., 9., 16.] == vec_bin_op(&m, &m, |x, y| x * y))
}

#[test]
fn frobenius_norm_test() {
    assert!((30f64).sqrt() == frobenius_norm(&vec![1.0, 2.0, 3.0, 4.0].to_matrix_2d().unwrap()));
}
