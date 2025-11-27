use std::{
   borrow::Borrow,
   fmt,
   hash::{Hash, Hasher},
   ops::{Deref, RangeBounds},
   str::Utf8Error,
};

use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Default)]
pub struct Str(Bytes);

impl Str {
   #[inline]
   pub fn from_string(s: String) -> Self {
      Self(Bytes::from(s.into_bytes()))
   }

   #[inline]
   pub fn copy_from_str(s: &str) -> Self {
      Self(Bytes::copy_from_slice(s.as_bytes()))
   }

   #[inline]
   pub fn from_static(s: &'static str) -> Self {
      Self(Bytes::from_static(s.as_bytes()))
   }

   #[inline]
   pub fn from_bytes(bytes: Bytes) -> Result<Self, Utf8Error> {
      std::str::from_utf8(&bytes)?;
      Ok(Self(bytes))
   }

   #[inline]
   pub fn slice_ref(&self, slice: &str) -> Self {
      Self(self.0.slice_ref(slice.as_bytes()))
   }

   #[inline]
   pub fn from_utf8_lossy(bytes: &[u8]) -> Self {
      let s = String::from_utf8_lossy(bytes);
      Self::from_string(s.to_string())
   }

   #[inline]
   pub fn as_str(&self) -> &str {
      // SAFETY: validated UTF-8 on construction (from_string, from_static,
      // from_bytes)
      unsafe { std::str::from_utf8_unchecked(&self.0) }
   }

   #[inline]
   #[must_use]
   pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
      Self(self.0.slice(range))
   }

   #[inline]
   #[must_use]
   pub fn trim(&self) -> Self {
      let s = self.as_str();
      let trimmed = s.trim();
      if trimmed.len() == s.len() {
         return self.clone();
      }
      let start = s.as_ptr() as usize - self.0.as_ptr() as usize + (s.len() - s.trim_start().len());
      let end = start + trimmed.len();
      self.slice(start..end)
   }

   #[inline]
   #[must_use]
   pub fn trim_start(&self) -> Self {
      let s = self.as_str();
      let trimmed = s.trim_start();
      if trimmed.len() == s.len() {
         return self.clone();
      }
      let start = s.len() - trimmed.len();
      self.slice(start..)
   }

   #[inline]
   #[must_use]
   pub fn trim_end(&self) -> Self {
      let s = self.as_str();
      let trimmed = s.trim_end();
      if trimmed.len() == s.len() {
         return self.clone();
      }
      self.slice(..trimmed.len())
   }

   #[inline]
   pub fn is_empty(&self) -> bool {
      self.0.is_empty()
   }

   #[inline]
   pub fn len(&self) -> usize {
      self.0.len()
   }

   #[inline]
   pub fn into_string(self) -> String {
      String::from_utf8(self.0.into()).expect("Str is always valid UTF-8")
   }
}

impl Deref for Str {
   type Target = str;

   #[inline]
   fn deref(&self) -> &str {
      self.as_str()
   }
}

impl AsRef<str> for Str {
   #[inline]
   fn as_ref(&self) -> &str {
      self.as_str()
   }
}

impl Borrow<str> for Str {
   #[inline]
   fn borrow(&self) -> &str {
      self.as_str()
   }
}

impl fmt::Debug for Str {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      fmt::Debug::fmt(self.as_str(), f)
   }
}

impl fmt::Display for Str {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      fmt::Display::fmt(self.as_str(), f)
   }
}

impl PartialEq for Str {
   #[inline]
   fn eq(&self, other: &Self) -> bool {
      self.as_str() == other.as_str()
   }
}

impl Eq for Str {}

impl PartialEq<str> for Str {
   #[inline]
   fn eq(&self, other: &str) -> bool {
      self.as_str() == other
   }
}

impl PartialEq<&str> for Str {
   #[inline]
   fn eq(&self, other: &&str) -> bool {
      self.as_str() == *other
   }
}

impl PartialEq<String> for Str {
   #[inline]
   fn eq(&self, other: &String) -> bool {
      self.as_str() == other.as_str()
   }
}

impl Hash for Str {
   #[inline]
   fn hash<H: Hasher>(&self, state: &mut H) {
      self.as_str().hash(state);
   }
}

impl From<String> for Str {
   #[inline]
   fn from(s: String) -> Self {
      Self::from_string(s)
   }
}

impl From<&'static str> for Str {
   #[inline]
   fn from(s: &'static str) -> Self {
      Self::from_static(s)
   }
}

impl From<Str> for String {
   #[inline]
   fn from(s: Str) -> Self {
      s.into_string()
   }
}

impl Serialize for Str {
   fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
      serializer.serialize_str(self.as_str())
   }
}

impl<'de> Deserialize<'de> for Str {
   fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
      let s = String::deserialize(deserializer)?;
      Ok(Self::from_string(s))
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn test_basic_operations() {
      let s = Str::from_string("hello world".to_string());
      assert_eq!(s.as_str(), "hello world");
      assert_eq!(s.len(), 11);
      assert!(!s.is_empty());
   }

   #[test]
   fn test_slicing() {
      let s = Str::from_string("hello world".to_string());
      let slice = s.slice(0..5);
      assert_eq!(slice.as_str(), "hello");

      let slice2 = s.slice(6..);
      assert_eq!(slice2.as_str(), "world");
   }

   #[test]
   fn test_trim() {
      let s = Str::from_string("  hello  ".to_string());
      assert_eq!(s.trim().as_str(), "hello");
      assert_eq!(s.trim_start().as_str(), "hello  ");
      assert_eq!(s.trim_end().as_str(), "  hello");
   }

   #[test]
   fn test_zero_copy() {
      let s = Str::from_string("hello world".to_string());
      let s2 = s.clone();
      let slice = s.slice(0..5);

      assert_eq!(s.0.as_ptr(), s2.0.as_ptr());
      assert_eq!(s.0.as_ptr(), slice.0.as_ptr());
   }
}
