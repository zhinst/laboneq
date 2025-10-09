// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use string_interner::symbol::SymbolU32;
use string_interner::{DefaultBackend, StringInterner};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamedId {
    uid: SymbolU32,
}

/// A store for named IDs
#[derive(Default)]
pub struct NamedIdStore {
    interner: StringInterner<DefaultBackend>,
}

impl NamedIdStore {
    pub fn new() -> Self {
        NamedIdStore {
            interner: StringInterner::new(),
        }
    }

    /// Return a unique ID for a given name, otherwise None.
    pub fn get(&self, name: impl AsRef<str>) -> Option<NamedId> {
        self.interner.get(name).map(|uid| NamedId { uid })
    }

    /// Return a unique ID for a given name, inserting it if necessary.
    pub fn get_or_insert(&mut self, name: impl AsRef<str>) -> NamedId {
        let uid = self.interner.get_or_intern(name);
        NamedId { uid }
    }

    /// Resolve a unique ID to its original string.
    pub fn resolve(&self, uid: impl Into<NamedId>) -> Option<&str> {
        self.interner.resolve(Into::<NamedId>::into(uid).uid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_id_store() {
        let mut store = NamedIdStore::new();
        let id1 = store.get_or_insert("test");
        let id2 = store.get_or_insert("test");
        assert_eq!(id1, id2);
        assert_eq!(store.resolve(id1), Some("test"));

        let id3 = store.get_or_insert("test2");
        let id4 = store.get_or_insert("test2");
        assert_ne!(id1, id3);
        assert_eq!(id3, id4);
        assert_eq!(store.resolve(id3), Some("test2"));

        let id5 = store.get("test2").unwrap();
        let id6 = store.get("test2").unwrap();
        assert_ne!(id1, id5);
        assert_eq!(id5, id6);
        assert_eq!(store.resolve(id5), Some("test2"));
    }
}
