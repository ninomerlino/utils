//! In-place sorting algorithms.
//!
//! The public sorting functions operate on mutable slices and support both
//! ascending and descending order through [`Order`]. They require `Copy` and
//! ordering traits because the implementations compare and move elements.

use crate::rng::{PRNG, Xoshiro256};

/// The requested ordering for a sorting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Smallest elements first.
    Ascending,
    /// Largest elements first.
    Descending,
}

fn shift_down<T: Copy + PartialOrd + PartialEq>(data: &mut [T], root: usize) {
    let mut root = root;
    loop {
        let left = root * 2 + 1;

        if left >= data.len() {
            break;
        }

        let right = left + 1;
        let mut biggest = left;

        if right < data.len() && data[right] > data[left] {
            biggest = right;
        }

        if data[root] >= data[biggest] {
            break;
        }

        data.swap(root, biggest);
        root = biggest;
    }
}

/// Sorts `data` in place using heapsort.
///
/// The algorithm runs in `O(n log n)` time and uses `O(1)` auxiliary space.
/// It is not stable.
pub fn heap_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], order: Order) {
    if data.len() == 0 || data.len() == 1 {
        return;
    }

    for root in (0..data.len() / 2).rev() {
        shift_down(data, root);
    }

    for end in (1..data.len()).rev() {
        data.swap(0, end);
        shift_down(&mut data[..end], 0);
    }

    if order == Order::Descending {
        data.reverse();
    }
}

/// Sorts `data` in place using selection sort.
///
/// Selection sort runs in `O(n²)` time and uses `O(1)` auxiliary space. It is
/// not stable in this implementation.
pub fn selection_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], order: Order) {
    if data.len() == 0 || data.len() == 1 {
        return;
    }

    for pos in 0..(data.len() - 1) {
        let mut min_pos = pos;
        for cur in pos + 1..data.len() {
            if (data[cur] < data[min_pos]) {
                min_pos = cur;
            }
        }
        data.swap(pos, min_pos);
    }
    if order == Order::Descending {
        data.reverse();
    }
}

fn _insertion_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T]) {
    if data.len() == 0 || data.len() == 1 {
        return;
    }

    for pos in 1..data.len() {
        let mut key = pos;
        let stored = data[key];

        while key >= 1 && stored < data[key - 1] {
            key -= 1;
        }

        data.copy_within(key..pos, key + 1);
        data[key] = stored;
    }
}

/// Sorts `data` in place using insertion sort.
///
/// Insertion sort runs in `O(n)` time for already sorted input and `O(n²)` in
/// the worst case. It uses `O(1)` auxiliary space and is stable for equal
/// elements.
pub fn insertion_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], order: Order) {
    _insertion_sort(data);
    if order == Order::Descending {
        data.reverse();
    }
}

fn pivot_data<T: Copy + PartialOrd + PartialEq>(
    data: &mut [T],
    rng: &mut impl PRNG,
) -> (usize, usize) {
    let random = rng.random_pick(data).unwrap();
    let pivot = data[random];

    let mut low = 0;
    let mut mid = 0;
    let mut high = data.len();

    while mid < high {
        if data[mid] < pivot {
            data.swap(low, mid);
            low += 1;
            mid += 1;
        } else if data[mid] > pivot {
            high -= 1;
            data.swap(mid, high);
        } else {
            mid += 1;
        }
    }

    (low, high)
}

fn _quick_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], rng: &mut impl PRNG) {
    match data.len() {
        0 | 1 => {}
        2 => {
            if data[0] > data[1] {
                data.swap(0, 1);
            }
        }
        _ => {
            let (low, high) = pivot_data(data, rng);
            _quick_sort(&mut data[..low], rng);
            _quick_sort(&mut data[high..], rng);
        }
    }
}

/// Sorts `data` in place using randomized three-way quicksort.
///
/// Average-case complexity is `O(n log n)` and worst-case complexity is
/// `O(n²)`. The randomized pivot and three-way partition help with sorted and
/// duplicate-heavy inputs, but the algorithm is not stable.
pub fn quick_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], order: Order) {
    let mut rng = Xoshiro256::default();
    _quick_sort(data, &mut rng);
    if order == Order::Descending {
        data.reverse();
    }
}

fn _intro_sort<T: Copy + PartialOrd + PartialEq>(
    data: &mut [T],
    threshold: usize,
    recursion_limit: usize,
    rng: &mut impl PRNG,
) {
    if data.len() <= 1 {
        return;
    }

    if data.len() <= threshold {
        insertion_sort(data, Order::Ascending);
        return;
    }

    if recursion_limit == 0 {
        heap_sort(data, Order::Ascending);
        return;
    }

    let (low, high) = pivot_data(data, rng);

    _intro_sort(&mut data[..low], threshold, recursion_limit - 1, rng);

    _intro_sort(&mut data[high..], threshold, recursion_limit - 1, rng);
}

/// Sorts `data` in place using introsort.
///
/// Introsort starts with randomized quicksort, uses insertion sort for small
/// partitions, and falls back to heapsort when recursion becomes too deep.
/// Its worst-case complexity is `O(n log n)` and it uses `O(log n)` stack
/// space. It is not stable.
pub fn intro_sort<T: Copy + PartialOrd + PartialEq>(data: &mut [T], order: Order) {
    if data.len() <= 1 {
        return;
    }
    let mut rng = Xoshiro256::default();
    let threshold = usize::BITS as usize;
    let recursion_limit = 2 * data.len().ilog2() as usize;
    _intro_sort(data, threshold, recursion_limit, &mut rng);
    if order == Order::Descending {
        data.reverse();
    }
}
