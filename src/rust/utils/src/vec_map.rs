// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::slice;
use std::{mem, vec};

/// An associative container backed by a sorted [`Vec`]. Insertions and lookups use binary search.
///
/// A [VecMap] is thus suitable when the number of stored items is small to medium-sized,
/// and lookup is much more frequent than insertion.
/// Keys are kept sorted, and iteration order follows the natural ordering of keys.
/// Insertion is O(N), lookup is O(log(N)).
#[derive(Clone, Default, Debug, PartialEq)]
pub struct VecMap<K, V>(Vec<(K, V)>);

impl<K, V> VecMap<K, V> {
    pub fn new() -> VecMap<K, V> {
        VecMap(Vec::new())
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.iter().map(|(k, _)| k)
    }

    pub fn keys_mut(&mut self) -> impl Iterator<Item = &mut K> {
        self.0.iter_mut().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.0.iter_mut().map(|(_, v)| v)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (K, V)> {
        self.0.iter_mut()
    }

    /// Insert the given value. Returns the old value for the same key.
    pub fn insert(&mut self, k: K, mut v: V) -> Option<V>
    where
        K: Ord,
    {
        match self.0.binary_search_by(|(probe, _)| probe.cmp(&k)) {
            Ok(idx) => {
                mem::swap(&mut self.0[idx].1, &mut v);
                Some(v)
            }
            Err(idx) => {
                self.0.insert(idx, (k, v));
                None
            }
        }
    }

    pub fn get(&self, key: &K) -> Option<&V>
    where
        K: Ord,
    {
        self.0
            .binary_search_by(|(probe, _)| probe.cmp(key))
            .ok()
            .map(|idx| &self.0[idx].1)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = (K, V)>)
    where
        K: Ord,
    {
        self.0.extend(iter);
        self.0.sort_by(|a, b| a.0.cmp(&b.0));
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&(K, V)) -> bool,
    {
        self.0.retain(|elem| f(elem));
    }
}

impl<K, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a VecMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = slice::Iter<'a, (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut VecMap<K, V> {
    type Item = &'a mut (K, V);
    type IntoIter = slice::IterMut<'a, (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for VecMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut result = VecMap::new();

        for (k, v) in iter.into_iter() {
            result.insert(k, v);
        }
        result
    }
}

#[macro_export]
macro_rules! vecmap {
    ($($x:expr),* $(,)?) => {
        utils::vec_map::VecMap::from_iter([$($x),*])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let map: VecMap<i32, String> = VecMap::new();
        assert_eq!(map.keys().count(), 0);
    }

    #[test]
    fn test_insert_new_key() {
        let mut map = VecMap::new();
        let result = map.insert("key1", "value1");
        assert_eq!(result, None);
        assert_eq!(map.keys().count(), 1);
    }

    #[test]
    fn test_insert_duplicate_key() {
        let mut map = VecMap::new();
        map.insert("key1", "value1");
        let result = map.insert("key1", "value2");
        assert_eq!(result, Some("value1"));
        assert_eq!(map.keys().count(), 1);
        assert_eq!(map.get(&"key1"), Some(&"value2"));
    }

    #[test]
    fn test_get_existing_key() {
        let mut map = VecMap::new();
        map.insert(42, "answer");
        assert_eq!(map.get(&42), Some(&"answer"));
    }

    #[test]
    fn test_get_nonexistent_key() {
        let mut map = VecMap::new();
        map.insert(42, "answer");
        assert_eq!(map.get(&43), None);
    }

    #[test]
    fn test_keys() {
        let mut map = VecMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_keys_mut() {
        let mut map = VecMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        for key in map.keys_mut() {
            *key *= 10;
        }
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec![10, 20]);
    }

    #[test]
    fn test_values() {
        let mut map = VecMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        let values: Vec<_> = map.values().copied().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_values_mut() {
        let mut map = VecMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        for value in map.values_mut() {
            *value *= 10;
        }
        let values: Vec<_> = map.values().copied().collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn test_iter() {
        let mut map = VecMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        let items: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(items, vec![("a", 1), ("b", 2)]);
    }

    #[test]
    fn test_iter_mut() {
        let mut map = VecMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        for (k, v) in map.iter_mut() {
            *v = (*k).len() as i32;
        }
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&1));
    }

    #[test]
    fn test_into_iter() {
        let mut map = VecMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        let items: Vec<_> = map.into_iter().collect();
        assert_eq!(items, vec![(1, "one"), (2, "two")]);
    }

    #[test]
    fn test_into_iter_ref() {
        let mut map = VecMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        let items: Vec<_> = (&map).into_iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(items, vec![(1, "one"), (2, "two")]);
        // map is still usable
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn test_into_iter_mut() {
        let mut map = VecMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        for (k, v) in &mut map {
            *v = *k * 100;
        }
        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&200));
    }

    #[test]
    fn test_empty_map_operations() {
        let map: VecMap<i32, i32> = VecMap::new();
        assert_eq!(map.keys().count(), 0);
        assert_eq!(map.values().count(), 0);
        assert_eq!(map.iter().count(), 0);
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn test_multiple_inserts_and_gets() {
        let mut map = VecMap::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
        assert_eq!(map.keys().count(), 10);
    }
}
