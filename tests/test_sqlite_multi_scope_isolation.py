"""Guard for multi-scope sqlite model isolation.

History: `build_sqlite_table_model` used to share `sa_column` Column objects
and `__table_args__` Index objects across every scoped derivation.
SQLAlchemy binds each to exactly one Table, so a second scope failed with:

  sqlalchemy.exc.ArgumentError: Column object 'url' already assigned to
  Table 'memu_resources'

Production uses one scope (STUserModel) so it never fired live. But the
memu test suite hid 7 failures + 16 errors because test modules defined
different scope classes. Commit 2f74fa3 deep-copies sa_columns + table_args
per scoped build. This test pins the fix — importing two different scope
classes and building models for both must not raise.
"""

from __future__ import annotations

from pydantic import BaseModel

from memu.database.sqlite.schema import get_sqlite_sqlalchemy_models


class _ScopeA(BaseModel):
    user_id: str | None = None


class _ScopeB(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def test_two_different_scopes_can_build_without_column_collision() -> None:
    models_a = get_sqlite_sqlalchemy_models(scope_model=_ScopeA)
    models_b = get_sqlite_sqlalchemy_models(scope_model=_ScopeB)

    # Each scoped derivation has its own Table instance.
    assert models_a.Resource.__table__ is not models_b.Resource.__table__
    assert models_a.MemoryItem.__table__ is not models_b.MemoryItem.__table__
    assert models_a.Entity.__table__ is not models_b.Entity.__table__
    assert models_a.Triple.__table__ is not models_b.Triple.__table__

    # Columns are different objects (deep-copied), each bound to its own table.
    url_a = models_a.Resource.__table__.c.url
    url_b = models_b.Resource.__table__.c.url
    assert url_a is not url_b
    assert url_a.table is models_a.Resource.__table__
    assert url_b.table is models_b.Resource.__table__


def test_same_scope_twice_returns_cached_models() -> None:
    first = get_sqlite_sqlalchemy_models(scope_model=_ScopeA)
    second = get_sqlite_sqlalchemy_models(scope_model=_ScopeA)
    assert first is second, "identical scope class should hit the _MODEL_CACHE"
