import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import ApprovedDomain, BlockedDomain
from rag.web.domain_policy import DomainPolicy


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


def test_domain_not_listed_is_denied_by_default(db_session):
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("https://random-blog.example/post")
    assert not decision.allowed
    assert decision.reason == "domain_not_in_approved_list"


def test_approved_domain_is_allowed(db_session):
    db_session.add(ApprovedDomain(domain="oer.example.edu", priority=100, min_quality_score=0.5))
    db_session.flush()
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("https://oer.example.edu/physics/newtons-laws")
    assert decision.allowed
    assert decision.priority == 100
    assert decision.min_quality_score == 0.5


def test_blocked_domain_overrides_approved(db_session):
    db_session.add(ApprovedDomain(domain="sketchy.example"))
    db_session.add(BlockedDomain(domain="sketchy.example", reason="known to host unreliable content"))
    db_session.flush()
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("https://sketchy.example/article")
    assert not decision.allowed
    assert "domain_blocked" in decision.reason


def test_blocked_domain_alone_is_denied(db_session):
    db_session.add(BlockedDomain(domain="bad.example", reason="spam"))
    db_session.flush()
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("https://bad.example/x")
    assert not decision.allowed


def test_subdomain_is_not_implicitly_approved(db_session):
    db_session.add(ApprovedDomain(domain="university.edu"))
    db_session.flush()
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("https://evil.university.edu.attacker.com/phish")
    assert not decision.allowed


def test_unparseable_url_is_denied(db_session):
    policy = DomainPolicy(db_session)
    decision = policy.evaluate("not a url at all")
    assert not decision.allowed
    assert decision.reason == "unparseable_url"
