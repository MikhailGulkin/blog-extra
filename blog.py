import asyncio
# domain

## entity
### root
from abc import ABC
from dataclasses import dataclass, field

from src.domain.common.events.event import Event

from .entity import Entity


@dataclass
class AggregateRoot(Entity, ABC):
    _events: list[Event] = field(default_factory=list, init=False, repr=False, hash=False, compare=False)

    def record_event(self, event: Event) -> None:
        self._events.append(event)

    def get_events(self) -> list[Event]:
        return self._events

    def clear_events(self) -> None:
        self._events.clear()

    def pull_events(self) -> list[Event]:
        events = self.get_events().copy()
        self.clear_events()
        return events
# entity
from abc import ABC


class Entity(ABC):
    pass
# vo base

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=Any)


@dataclass(frozen=True)
class BaseValueObject(ABC):
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """This method checks that a value is valid to create this value object"""
        pass


@dataclass(frozen=True)
class ValueObject(BaseValueObject, Generic[T]):
    value: T

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        return int(self.value)

# validate
from dataclasses import dataclass
from uuid import UUID



@dataclass(eq=False)
class UserIsDeleted(RuntimeError, DomainException):
    user_id: UUID

    @property
    def message(self) -> str:
        return f'The user with "{self.user_id}" user_id is deleted'

# vo
from dataclasses import dataclass



@dataclass(frozen=True)
class UserEmail(ValueObject[str]):
    value: str
from dataclasses import dataclass, field
from uuid import UUID, uuid4



@dataclass(frozen=True)
class UserId(ValueObject[UUID]):
    value: UUID = field(default_factory=uuid4)

    @property
    def to_uuid(self) -> UUID:
        return self.value
from dataclasses import dataclass



@dataclass(frozen=True)
class UserPassword(ValueObject[str]):
    value: str
from dataclasses import dataclass



@dataclass(frozen=True)
class UserName(ValueObject[str]):
    value: str


# user
import dataclasses
from dataclasses import dataclass




@dataclass
class User(AggregateRoot):
    id: UserId
    username: UserName
    password: UserPassword
    email: UserEmail
    deleted: bool = dataclasses.field(default=False, kw_only=True)

    @classmethod
    def create(
        cls,
        username: UserName,
        password: UserPassword,
        email: UserEmail,
        user_id: UserId = UserId(),
    ) -> "User":
        user = User(id=user_id, username=username, password=password, email=email)
        user.record_event(
            UserCreated(
                id=user_id.to_uuid,
                username=str(username),
                password=str(password),
                email=str(email),
            )
        )
        return user

    def update(
        self,
        username: UserName | Empty = Empty.UNSET,
        email: UserEmail | Empty = Empty.UNSET,
        password: UserPassword | Empty = Empty.UNSET,
    ) -> None:
        if username is not Empty.UNSET:
            self.username = username
        if email is not Empty.UNSET:
            self.email = email
        if password is not Empty.UNSET:
            self.password = password

# application
# common
###uow

from abc import ABC, abstractmethod


class UnitOfWork(ABC):
    @abstractmethod
    async def commit(self) -> None:
        pass

    @abstractmethod
    async def rollback(self) -> None:
        pass

###mapper
from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")


class Mapper(ABC):
    @abstractmethod
    def load(self, data: Any, class_: type[T]) -> T:
        pass
###command
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import didiator

CRes = TypeVar("CRes")


class Command(didiator.Command[CRes], ABC, Generic[CRes]):
    @abstractmethod
    def validate(self) -> None:
        pass


C = TypeVar("C", bound=Command)


class CommandHandler(didiator.CommandHandler[C, CRes], ABC, Generic[C, CRes]):
    pass
### event
from abc import ABC
from typing import Generic, TypeVar

import didiator
from src.domain.common.events.event import Event

E = TypeVar("E", bound=Event)


class EventHandler(didiator.EventHandler[E], ABC, Generic[E]):
    pass

from dataclasses import dataclass

from didiator import EventMediator


@dataclass(frozen=True)
class CreateUser(Command[dto.User]):
    username: str
    password: str
    email: str

    def validate(self) -> None:
        validators.validate_username(self.username)
        validators.validate_email(self.email)
        validators.validate_password(self.password)


class CreateUserHandler(CommandHandler[CreateUser, dto.User]):
    def __init__(self, user_repo: UserRepo, uow: UnitOfWork, mapper: Mapper, mediator: EventMediator) -> None:
        self._user_repo = user_repo
        self._uow = uow
        self._mapper = mapper
        self._mediator = mediator

    async def __call__(self, command: CreateUser) -> dto.User:
        user = User.create(
            username=UserName(command.username),
            email=UserEmail(command.email),
            password=UserPassword(command.password),
        )

        await self._user_repo.add_user(user)
        await self._mediator.publish(user.pull_events())
        await self._uow.commit()

        user_dto = self._mapper.load(user, dto.User)
        return user_dto

# db

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import DBConfig


async def build_sa_engine(db_config: DBConfig) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(
        db_config.full_url,
    )
    yield engine

    await engine.dispose()


def build_sa_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    session_factory = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return session_factory


async def build_sa_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
# model
from uuid import UUID, uuid4

from sqlalchemy import False_
from sqlalchemy.orm import Mapped, mapped_column

from .base import TimedBaseModel


class User(TimedBaseModel):
    __tablename__ = "users"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    deleted: Mapped[bool] = mapped_column(default=False, server_default=False_())


# Repo
from typing import NoReturn
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError
from src.application.common.exceptions import RepoError
from src.application.user import dto
from src.application.user.exceptions import (
    UserEmailAlreadyExists,
    UserIdAlreadyExists,
    UserIdNotExist,
    UserNameAlreadyExists,
    UserNameNotExist,
)
from src.application.user.interfaces.persistence import UserReader, UserRepo
from src.domain.common.constants import Empty
from src.domain.user import entities
from src.domain.user.value_objects import UserId
from src.infrastructure.db.exception_mapper import exception_mapper
from src.infrastructure.db.models.user import User
from src.infrastructure.db.repositories.base import SQLAlchemyRepo


class UserReaderImpl(SQLAlchemyRepo, UserReader):
    @exception_mapper
    async def get_user_by_id(self, user_id: UUID) -> dto.UserDTOs:
        user = await self.session.scalar(
            select(User).where(
                User.id == user_id,
            )
        )
        if user is None:
            raise UserIdNotExist(user_id)

        return self._mapper.load(user, dto.UserDTOs)

    @exception_mapper
    async def get_user_by_username(self, username: str) -> dto.User:
        user = await self.session.scalar(
            select(User).where(
                User.username == username,
            )
        )
        if user is None:
            raise UserNameNotExist(username)

        return self._mapper.load(user, dto.User)

    @exception_mapper
    async def get_users_count(self, deleted: bool | Empty = Empty.UNSET) -> int:
        query = select(func.count(User.id))

        if deleted is not Empty.UNSET:
            query = query.where(User.deleted == deleted)

        users_count = await self.session.scalar(query)
        return users_count or 0


class UserRepoImpl(SQLAlchemyRepo, UserRepo):
    @exception_mapper
    async def acquire_user_by_id(self, user_id: UserId) -> entities.User:
        user = await self.session.scalar(
            select(User)
            .where(
                User.id == user_id.to_uuid,
            )
            .with_for_update()
        )

        if user is None:
            raise UserIdNotExist(user_id.to_uuid)

        return self._mapper.load(user, entities.User)

    @exception_mapper
    async def add_user(self, user: entities.User) -> None:
        db_user = self._mapper.load(user, User)
        self.session.add(db_user)
        try:
            await self.session.flush((db_user,))
        except IntegrityError as err:
            self._parse_error(err, user)

    def _parse_error(self, err: DBAPIError, user: entities.User) -> NoReturn:
        match err.__cause__.__cause__.constraint_name:  # type: ignore
            case "pk_users":
                raise UserIdAlreadyExists(user.id.to_uuid) from err
            case "uq_users_username":
                raise UserNameAlreadyExists(str(user.username)) from err
            case "uq_users_email":
                raise UserEmailAlreadyExists(str(user.email)) from err
            case _:
                raise RepoError from err



#conrollers

from uuid import UUID

from didiator import CommandMediator, QueryMediator
from fastapi import APIRouter, Depends, Path


user_router = APIRouter(
    prefix="/users",
    tags=["users"],
)


@user_router.post("/", **user_create)
async def create_user(
    create_user_command: CreateUser,
    mediator: CommandMediator = Depends(Stub(CommandMediator)),
) -> dto.User:
    create_user_command.validate()

    user = await mediator.send(create_user_command)
    return user


@user_router.get("/{user_id}", **user_get_by_id)
async def get_user_by_id(
    user_id: UUID,
    mediator: QueryMediator = Depends(Stub(QueryMediator)),
) -> dto.UserDTOs:
    user = await mediator.query(GetUserById(user_id=user_id))
    return user


@user_router.get("/@/{username}", **user_get_by_username)
async def get_user_by_username(
    username: str = Path(max_length=MAX_USERNAME_LENGTH),
    mediator: QueryMediator = Depends(Stub(QueryMediator)),
) -> dto.User:
    user = await mediator.query(GetUserByUsername(username=username))
    return user


# config

@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = __debug__


@dataclass
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    api: APIConfig = field(default_factory=APIConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)


# init app

def init_api(
) -> FastAPI:
    app = FastAPI(debug=False, title="User service", version="1.0.0", default_response_class=ORJSONResponse)
    setup_providers(app, mediator, mapper, di_builder, di_state)
    setup_middlewares(app)
    setup_controllers(app)
    return app


async def run_api(app: FastAPI, api_config: APIConfig) -> None:
    config = uvicorn.Config(app, host=api_config.host, port=api_config.port)
    server = uvicorn.Server(config)

    await server.serve()


if __name__ == '__main__':
    asyncio.run(run_api(init_api(), Config.api))