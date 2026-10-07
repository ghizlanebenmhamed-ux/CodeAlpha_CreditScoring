import unittest
from unittest.mock import Mock
import numpy as np
from model import CreditModel,FIELDS
from app import CreditApp

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=CreditModel()
    def values(self,index):return self.model.frame.iloc[index][FIELDS].to_dict()
    def test_split(self):
        self.assertFalse(set(self.model.train_ids)&set(self.model.test_ids))
        self.assertEqual(len(self.model.train_ids),800);self.assertEqual(len(self.model.test_ids),200)
    def test_scaler_training_only(self):
        engineered=self.model.engineer(self.model.frame[FIELDS]).iloc[self.model.train_ids]
        from model import NUMERIC
        np.testing.assert_allclose(self.model.pipeline[0].named_transformers_['numbers'].mean_,engineered[NUMERIC+['log_amount','monthly_principal']].mean())
    def test_all_test_predictions(self):
        for i in self.model.test_ids:
            label,score=self.model.predict(self.values(i));self.assertIn(label,['good','bad']);self.assertTrue(0<=score<=1)
    def test_engineering(self):
        data=self.model.engineer(self.model.frame[FIELDS])
        np.testing.assert_allclose(data.monthly_principal,data.amount/data.duration)
    def test_invalid_input(self):
        for field,value in [('duration',0),('amount',float('nan')),('installment_rate',5),('history','wrong'),('existing_credits',1.5)]:
            data=self.values(self.model.test_ids[0]);data[field]=value
            with self.assertRaises(ValueError):self.model.predict(data)
    def app(self):
        app=CreditApp.__new__(CreditApp);app.model=self.model;app.current_id=int(self.model.test_ids[0]);app.inputs={};app.result=Mock()
        for key,value in self.values(app.current_id).items():
            var=Mock();var.get.return_value=str(value);app.inputs[key]=var
        return app
    def test_predict_handler(self):
        app=self.app();app.predict();self.assertIn('Recorded label',app.result.set.call_args.args[0])
    def test_enter(self):
        app=self.app();app.predict(Mock());self.assertIn('Predicted dataset class',app.result.set.call_args.args[0])
    def test_input_error_handler(self):
        app=self.app();app.inputs['amount'].get.return_value='abc';app.predict();self.assertIn('Input error',app.result.set.call_args.args[0])
    def test_clear(self):
        app=self.app();app.clear();self.assertIsNone(app.current_id)
        for var in app.inputs.values():var.set.assert_called_once_with('')
    def test_edit_removes_recorded_label(self):
        app=self.app();app.changed();self.assertIsNone(app.current_id)
    def test_metrics(self):
        self.assertEqual(sum(map(sum,self.model.metrics['confusion_matrix'])),200)
        self.assertGreater(self.model.metrics['accuracy'],self.model.metrics['majority_baseline_accuracy'])

if __name__=='__main__':unittest.main(verbosity=2)
