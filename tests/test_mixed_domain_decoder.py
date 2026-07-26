import unittest
from scripts.train_mixed_domain_decoder import balanced_epoch_order

class MixedDomainDecoderTests(unittest.TestCase):
    def test_balanced_order_alternates_domains(self):
        order=balanced_epoch_order(list(range(5)),list(range(2)),seed=1)
        self.assertEqual(len(order),10)
        self.assertTrue(all(order[i][0]!=order[i+1][0] for i in range(len(order)-1)))
        self.assertEqual(sum(d=="camri" for d,_ in order),sum(d=="mouse" for d,_ in order))

if __name__=="__main__":unittest.main()
