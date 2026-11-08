use std::{
    array::{self},
    fmt::Debug,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, Rem, RemAssign, Shl, ShlAssign,
        Shr, ShrAssign, Sub, SubAssign,
    },
};

/// Numeric operations required by [`Vector`] and [`super::Matrix`].
///
/// Implementations are provided for the built-in signed integer and floating
/// point types. Decompositions and least-squares operations are intended for
/// floating-point types; integer division and square roots truncate.
pub trait VectorType:
    Default
    + Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + PartialEq
    + PartialOrd
    + Debug
    + Neg<Output = Self>
{
    fn unit() -> Self;
    fn epsilon() -> Self;
    fn absolute_value(self) -> Self;
    fn approximate_zero(self) -> bool {
        self.absolute_value() <= Self::epsilon()
    }
    fn sign(self) -> Self;
    fn sqrt(self) -> Self;
}

impl VectorType for i8 {
    fn unit() -> Self {
        1
    }

    fn epsilon() -> Self {
        0
    }

    fn absolute_value(self) -> Self {
        self.abs()
    }

    fn approximate_zero(self) -> bool {
        self.absolute_value() <= Self::epsilon()
    }

    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        self.isqrt()
    }
}
impl VectorType for i16 {
    fn unit() -> Self {
        1
    }
    fn epsilon() -> Self {
        0
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }

    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        self.isqrt()
    }
}
impl VectorType for i32 {
    fn unit() -> Self {
        1
    }
    fn epsilon() -> Self {
        0
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }
    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        self.isqrt()
    }
}
impl VectorType for i64 {
    fn unit() -> Self {
        1
    }
    fn epsilon() -> Self {
        0
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }
    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        self.isqrt()
    }
}
impl VectorType for i128 {
    fn unit() -> Self {
        1
    }
    fn epsilon() -> Self {
        0
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }
    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        self.isqrt()
    }
}

impl VectorType for f32 {
    fn unit() -> Self {
        1.0
    }
    fn epsilon() -> Self {
        f32::EPSILON
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }
    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}
impl VectorType for f64 {
    fn unit() -> Self {
        1.0
    }
    fn epsilon() -> Self {
        f64::EPSILON
    }
    fn absolute_value(self) -> Self {
        self.abs()
    }
    fn sign(self) -> Self {
        self.signum()
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq)]
/// A fixed-size vector containing `SIZE` values of type `Type`.
pub struct Vector<Type: VectorType, const SIZE: usize> {
    values: [Type; SIZE],
}

impl<T: VectorType, const S: usize> Vector<T, S> {
    /// Creates a vector whose entries are all the numeric unit value.
    pub fn unit() -> Self {
        Self::from(T::unit())
    }

    /// Returns the vector's compile-time length, `S`.
    pub fn len(&self) -> usize {
        S
    }

    pub fn is_empty(&self) -> bool {
        S == 0
    }

    /// Iterates over the entries by shared reference.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    /// Iterates over the entries by mutable reference.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.values.iter_mut()
    }

    /// Returns the sum of the entries.
    pub fn sum(&self) -> T {
        self.values.iter().fold(T::default(), |acc, val| acc + *val)
    }

    /// Returns the squared Euclidean length, the sum of each entry squared.
    pub fn magnitude_squared(&self) -> T {
        self.values
            .iter()
            .fold(T::default(), |acc, &val| (val * val) + acc)
    }

    /// Returns the dot product with `other`.
    pub fn dot(&self, other: &Self) -> T {
        (*self * *other).sum()
    }

    /// Returns the Manhattan distance, the sum of absolute coordinate
    /// differences. The historical method name contains a spelling error.
    pub fn manahattan_distance(&self, other: &Self) -> T {
        (*self - *other).abs().sum()
    }

    /// Counts the coordinates that differ between the two vectors.
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let zero = T::default();
        (*self - *other).iter().filter(|&&i| i != zero).count()
    }

    /// Returns the squared Euclidean distance between the vectors.
    pub fn squared_euclidian_distance(&self, other: &Self) -> T {
        (*self - *other).magnitude_squared()
    }

    /// Returns the greatest entry and its index, or `None` for an empty vector.
    pub fn max(&self) -> Option<(T, usize)> {
        let mut t: Option<(T, usize)> = Option::None;

        for (i, v) in self.iter().enumerate() {
            if let Some((max, _)) = t {
                if *v > max {
                    t = Some((*v, i));
                }
            } else {
                t = Some((*v, i));
            }
        }
        t
    }

    /// Returns the least entry and its index, or `None` for an empty vector.
    pub fn min(&self) -> Option<(T, usize)> {
        let mut t: Option<(T, usize)> = Option::None;

        for (i, v) in self.iter().enumerate() {
            if let Some((min, _)) = t {
                if *v < min {
                    t = Some((*v, i));
                }
            } else {
                t = Some((*v, i));
            }
        }
        t
    }

    /// Returns the arithmetic mean of the entries.
    pub fn avg(&self) -> T {
        let mut sum = T::default();
        let mut len = T::default();

        for v in self.iter() {
            sum += *v;
            len += T::unit();
        }
        sum / len
    }

    pub fn product(&self) -> T {
        self.values.iter().fold(T::unit(), |acc, &a| acc * a)
    }

    /// Reverses the order of the entries in place.
    pub fn reverse(&mut self) {
        self.values.reverse();
    }

    /// Returns a vector containing the absolute value of each entry.
    pub fn abs(&self) -> Self {
        let mut r = *self;
        for v in r.iter_mut() {
            *v = v.absolute_value();
        }
        r
    }

    /// Returns the Euclidean length of the vector.
    pub fn magnitude(&self) -> T {
        T::sqrt(self.magnitude_squared())
    }

    /// Returns this vector divided by its magnitude. A zero vector cannot be
    /// normalized.
    pub fn normalized(&self) -> Self {
        *self / self.magnitude()
    }

    /// Returns the cosine of the angle between this vector and `other`.
    pub fn cosine(&self, other: &Self) -> T {
        dbg!(self.dot(other)) / self.magnitude() / other.magnitude()
    }

    /// Returns the sine magnitude of the angle between this vector and `other`.
    pub fn sine(&self, other: &Self) -> T {
        let cosine = self.cosine(other);
        T::sqrt(T::unit() - cosine * cosine)
    }

    /// Returns the Euclidean distance between the vectors.
    pub fn euclidian_distance(&self, other: &Self) -> T {
        (*self - *other).magnitude()
    }

    /// Returns the coordinate-wise reciprocal. Zero entries divide by zero.
    pub fn reciprocal(&self) -> Self {
        Self::unit() / *self
    }

    /// Returns the population standard deviation of the entries.
    pub fn std_deviation(&self) -> T {
        let mean = self.avg();
        let mut sum = T::default();
        for v in self.iter() {
            sum += (*v - mean) * (*v - mean);
        }
        T::sqrt(sum / T::unit())
    }
}

impl<T> Vector<T, 3>
where
    T: VectorType,
{
    /// Returns the three-dimensional cross product.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            values: [
                self[1] * other[2] - self[2] * other[1],
                self[2] * other[0] - self[0] * other[2],
                self[0] * other[1] - self[1] * other[0],
            ],
        }
    }
}

// ============================================================
// From Traits
// ============================================================

impl<T: VectorType, const S: usize> From<[T; S]> for Vector<T, S> {
    fn from(value: [T; S]) -> Self {
        Self { values: value }
    }
}

impl<T: VectorType, const S: usize> From<T> for Vector<T, S> {
    fn from(value: T) -> Self {
        Self { values: [value; S] }
    }
}

impl<T: VectorType, const S: usize> Default for Vector<T, S> {
    fn default() -> Self {
        Self {
            values: array::from_fn(|_| T::default()),
        }
    }
}

// ============================================================
// Index Traits
// ============================================================

impl<T: VectorType, const S: usize> Index<usize> for Vector<T, S> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T: VectorType, const S: usize> IndexMut<usize> for Vector<T, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T: VectorType, const S: usize> Index<Range<usize>> for Vector<T, S> {
    type Output = [T];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.values[range]
    }
}

impl<T: VectorType, const S: usize> IndexMut<Range<usize>> for Vector<T, S> {
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.values[range]
    }
}

// ============================================================
// <op> Vector
// ============================================================

impl<T: VectorType, const S: usize> Neg for Vector<T, S>
where
    T: Neg<Output = T>,
{
    type Output = Vector<T, S>;

    fn neg(self) -> Self::Output {
        let mut result = Self::default();

        for i in 0..S {
            result.values[i] = -self.values[i];
        }

        result
    }
}

// ============================================================
// Vector <op> Vector
// ============================================================

impl<T: VectorType, const S: usize> Add for Vector<T, S> {
    type Output = Vector<T, S>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] + rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Sub for Vector<T, S> {
    type Output = Vector<T, S>;

    fn sub(self, other: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] - other.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Mul for Vector<T, S> {
    type Output = Vector<T, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] * rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Div for Vector<T, S> {
    type Output = Vector<T, S>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] / rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Rem for Vector<T, S> {
    type Output = Vector<T, S>;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] % rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Shl for Vector<T, S>
where
    T: Shl<Output = T>,
{
    type Output = Vector<T, S>;

    fn shl(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] << rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> Shr for Vector<T, S>
where
    T: Shr<Output = T>,
{
    type Output = Vector<T, S>;

    fn shr(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] >> rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitAnd for Vector<T, S>
where
    T: BitAnd<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] & rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitOr for Vector<T, S>
where
    T: BitOr<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] | rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitXor for Vector<T, S>
where
    T: BitXor<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] ^ rhs.values[i]
        }
        r
    }
}

// ============================================================
// Vector <op>= Vector
// ============================================================

impl<T: VectorType, const S: usize> AddAssign for Vector<T, S> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] += rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> SubAssign for Vector<T, S> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] -= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> MulAssign for Vector<T, S> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] *= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> DivAssign for Vector<T, S> {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] /= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> RemAssign for Vector<T, S> {
    fn rem_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] %= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> ShlAssign for Vector<T, S>
where
    T: ShlAssign,
{
    fn shl_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] <<= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> ShrAssign for Vector<T, S>
where
    T: ShrAssign,
{
    fn shr_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] >>= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> BitAndAssign for Vector<T, S>
where
    T: BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] &= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> BitOrAssign for Vector<T, S>
where
    T: BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] |= rhs.values[i]
        }
    }
}

impl<T: VectorType, const S: usize> BitXorAssign for Vector<T, S>
where
    T: BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..S {
            self.values[i] ^= rhs.values[i]
        }
    }
}

// ============================================================
// Vector <op> Scalar (T)
// ============================================================

impl<T: VectorType, const S: usize> Add<T> for Vector<T, S> {
    type Output = Vector<T, S>;

    fn add(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] + rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Sub<T> for Vector<T, S> {
    type Output = Vector<T, S>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] - rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Mul<T> for Vector<T, S> {
    type Output = Vector<T, S>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] * rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Div<T> for Vector<T, S> {
    type Output = Vector<T, S>;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] / rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Rem<T> for Vector<T, S> {
    type Output = Vector<T, S>;

    fn rem(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] % rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Shl<T> for Vector<T, S>
where
    T: Shl<Output = T>,
{
    type Output = Vector<T, S>;

    fn shl(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] << rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> Shr<T> for Vector<T, S>
where
    T: Shr<Output = T>,
{
    type Output = Vector<T, S>;

    fn shr(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] >> rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitAnd<T> for Vector<T, S>
where
    T: BitAnd<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitand(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] & rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitOr<T> for Vector<T, S>
where
    T: BitOr<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitor(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] | rhs
        }
        r
    }
}

impl<T: VectorType, const S: usize> BitXor<T> for Vector<T, S>
where
    T: BitXor<Output = T>,
{
    type Output = Vector<T, S>;

    fn bitxor(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..S {
            r.values[i] = self.values[i] ^ rhs
        }
        r
    }
}

// ============================================================
// Vector <op>= Scalar (T)
// ============================================================

impl<T: VectorType, const S: usize> AddAssign<T> for Vector<T, S> {
    fn add_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] += rhs
        }
    }
}

impl<T: VectorType, const S: usize> SubAssign<T> for Vector<T, S> {
    fn sub_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] -= rhs
        }
    }
}

impl<T: VectorType, const S: usize> MulAssign<T> for Vector<T, S> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] *= rhs
        }
    }
}

impl<T: VectorType, const S: usize> DivAssign<T> for Vector<T, S> {
    fn div_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] /= rhs
        }
    }
}

impl<T: VectorType, const S: usize> RemAssign<T> for Vector<T, S> {
    fn rem_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] %= rhs
        }
    }
}

impl<T: VectorType, const S: usize> ShlAssign<T> for Vector<T, S>
where
    T: ShlAssign,
{
    fn shl_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] <<= rhs
        }
    }
}

impl<T: VectorType, const S: usize> ShrAssign<T> for Vector<T, S>
where
    T: ShrAssign,
{
    fn shr_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] >>= rhs
        }
    }
}

impl<T: VectorType, const S: usize> BitAndAssign<T> for Vector<T, S>
where
    T: BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] &= rhs
        }
    }
}

impl<T: VectorType, const S: usize> BitOrAssign<T> for Vector<T, S>
where
    T: BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] |= rhs
        }
    }
}

impl<T: VectorType, const S: usize> BitXorAssign<T> for Vector<T, S>
where
    T: BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: T) {
        for i in 0..S {
            self.values[i] ^= rhs
        }
    }
}

impl<'a, T: VectorType, const S: usize> IntoIterator for &'a Vector<T, S> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a, T: VectorType, const S: usize> IntoIterator for &'a mut Vector<T, S> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

impl<T: VectorType, const S: usize> IntoIterator for Vector<T, S> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}
