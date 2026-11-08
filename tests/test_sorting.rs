use utils::sorting::{
    Order, heap_sort, insertion_sort, intro_sort, quick_sort, selection_sort,
};

fn check_sort(mut data: Vec<i32>, order: Order, expected: Vec<i32>, sort: fn(&mut [i32], Order)) {
    sort(&mut data, order);
    assert_eq!(data, expected);
}

fn test_sort(sort: fn(&mut [i32], Order)) {
    check_sort(vec![3, 1, 4, 2], Order::Ascending, vec![1, 2, 3, 4], sort);
    check_sort(vec![3, 1, 4, 2], Order::Descending, vec![4, 3, 2, 1], sort);
    check_sort(vec![], Order::Ascending, vec![], sort);
    check_sort(vec![], Order::Descending, vec![], sort);
    check_sort(vec![1], Order::Ascending, vec![1], sort);
    check_sort(vec![1], Order::Descending, vec![1], sort);
}

#[test]
fn heap_sort_tests() {
    test_sort(heap_sort);
}

#[test]
fn selection_sort_tests() {
    test_sort(selection_sort);
}

#[test]
fn insertion_sort_tests() {
    test_sort(insertion_sort);
}

#[test]
fn quick_sort_tests() {
    test_sort(quick_sort);
}

#[test]
fn intro_sort_tests() {
    test_sort(intro_sort);
}
