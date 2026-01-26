from args import parse_args
from evolution import ape, ga_evo, de_evo
from utils import set_seed
from evaluator import CLSEvaluator, GENEvalutator
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai.chat_models import ChatOpenAI
from src.utils.load_dataset_coolprompt import load_dataset
from sklearn.model_selection import train_test_split


def run(args):
    set_seed(args.seed)
    task2evaluator = {
        "cls": CLSEvaluator,
        "gen": GENEvalutator,
    }
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2,  # 1 запрос в секунду
        check_every_n_seconds=0.1,  # проверять каждые 100ms
        max_bucket_size=10  # максимальный размер буфера
    )

    model = ChatOpenAI(
        # model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4000,
        timeout=60,
        max_retries=2,
        rate_limiter=rate_limiter
    )
    dataset, targets = load_dataset(args.dataset, size=40)
    train_dataset, valid_dataset, train_targets, valid_targets = train_test_split(
        dataset, targets, test_size=0.5
    )

    evaluator = task2evaluator[args.task](args, model, train_dataset, train_targets, valid_dataset, valid_targets)
    evaluator.logger.info(
        "---------------------Evolving prompt-------------------\n"
    )
    if args.evo_mode == "para" and args.para_mode == "topk":
        ape(args=args, evaluator=evaluator)
    elif args.evo_mode in 'de':
        de_evo(args=args, evaluator=evaluator)
    elif args.evo_mode == 'ga':
        ga_evo(args=args, evaluator=evaluator)

if __name__ == "__main__":
    args = parse_args()
    run(args)
