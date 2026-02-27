// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;

use crate::{ContextualError, LabOneQError, WithContext};

#[derive(thiserror::Error, Debug)]
#[error("{source}")]
pub struct ResourceExhaustionError {
    source: ContextualError,

    /// The usage of said exhausted resource, > 1.0, where 1.0 means total (100%) utilization
    pub usage: f64,
}

impl ResourceExhaustionError {
    pub fn new(msg: impl Into<Cow<'static, str>>, usage: f64) -> Self {
        assert!(usage > 1.0, "invalid usage report");
        ResourceExhaustionError {
            source: ContextualError::from_str(msg),
            usage,
        }
    }
}

impl WithContext for ResourceExhaustionError {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
    {
        self.source.add_context(f);
    }
}

// Collect the given iterator into Result<Vec<T>, LabOneQError> as you would expect from std collect,
// i.e. if all items are Ok(..), the collection result if Ok(Vec<T>), and if any item is Err(e), then
// the collection result is Err(e). The, only difference is that ResourceExhaustionErrors are treated
// differently - the first one encountered is not returned immediately, but all are seen and the one
// corresponding to the most severe exhaustion is returned.
pub fn intercept_and_collect<T>(
    items: impl Iterator<Item = Result<T, LabOneQError>>,
) -> Result<Vec<T>, LabOneQError> {
    let mut max_exhaustion: Option<ResourceExhaustionError> = None;
    let collected = items
        .filter_map(|item| match item {
            Err(LabOneQError::ResourceExhaustion(err)) => {
                match &max_exhaustion {
                    None => {
                        max_exhaustion = Some(err);
                    }
                    Some(prev) if err.usage > prev.usage => max_exhaustion = Some(err),
                    _ => {}
                }
                None
            }
            _ => Some(item),
        })
        .collect::<Result<_, LabOneQError>>()?;

    if let Some(err) = max_exhaustion {
        return Err(err.into());
    }

    Ok(collected)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_intercept_and_collect_ok() {
        let items = vec![Ok(1), Ok(2)];

        let res = intercept_and_collect(items.into_iter());
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_intercept_and_collect_err() {
        let items = vec![
            Err(ContextualError::from_str("abc").into()),
            Ok(1),
            Err(ContextualError::from_str("xyz").into()),
        ];
        let res = intercept_and_collect(items.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "abc");
    }

    #[test]
    fn test_intercept_and_collect_err_resource() {
        let items = vec![
            Err(ResourceExhaustionError::new("abc", 1.4).into()),
            Ok(1),
            Err(ResourceExhaustionError::new("xyz", 1.2).into()),
        ];
        let items_rev = vec![
            Err(ResourceExhaustionError::new("xyz", 1.2).into()),
            Ok(1),
            Err(ResourceExhaustionError::new("abc", 1.4).into()),
        ];

        let res = intercept_and_collect(items.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "abc");

        let res = intercept_and_collect(items_rev.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "abc");
    }

    #[test]
    fn test_intercept_and_collect_err_resource_no_usage() {
        let items = vec![Err(ResourceExhaustionError::new("abc", 1.1).into()), Ok(1)];

        let res = intercept_and_collect(items.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "abc");
    }

    #[test]
    fn test_intercept_mixed_errors() {
        // test that the first non-resource-exhaustion error is returned no matter
        // in which order it arrives relative to resource-exhaustion error
        let items_res_first = vec![
            Err(ResourceExhaustionError::new("resource error", 1.1).into()),
            Ok(1),
            Err(ContextualError::from_str("other error").into()),
        ];
        let items_other_first = vec![
            Err(ContextualError::from_str("other error").into()),
            Ok(1),
            Err(ResourceExhaustionError::new("resource error", 1.1).into()),
        ];

        let res = intercept_and_collect(items_res_first.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "other error");

        let res = intercept_and_collect(items_other_first.into_iter());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "other error");
    }
}
