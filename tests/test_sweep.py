from tomato_picovna import Sweep

if __name__ == "__main__":
    kwargs = {"start": 5_500_000_000, "stop": 7_500_000_000, "points": 10001}
    print(Sweep(**kwargs))

    kwargs = {"start": "5_500 MHz", "stop": "7.5 GHz", "points": 101}
    print(Sweep(**kwargs))
