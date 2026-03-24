import unittest

from data.medical_dataset import (
    MedicalSample,
    map_guangzhou_label,
    map_nih_label,
    split_samples_to_clients,
    stratified_split,
)


class MedicalDatasetTests(unittest.TestCase):
    def test_map_nih_label_keeps_only_binary_pneumonia_task(self):
        self.assertEqual(map_nih_label({"Finding Labels": "No Finding"}), 0)
        self.assertEqual(map_nih_label({"Finding Labels": "Pneumonia"}), 1)
        self.assertIsNone(map_nih_label({"Finding Labels": "Infiltration"}))

    def test_map_guangzhou_label_accepts_strings_and_ints(self):
        self.assertEqual(map_guangzhou_label({"label": "NORMAL"}), 0)
        self.assertEqual(map_guangzhou_label({"label": "PNEUMONIA"}), 1)
        self.assertEqual(map_guangzhou_label({"label": 1}), 1)

    def test_stratified_split_preserves_both_classes(self):
        samples = [
            MedicalSample("s", idx, idx % 2, "nih")
            for idx in range(20)
        ]
        train_samples, val_samples = stratified_split(samples, val_ratio=0.2, seed=7)
        self.assertTrue(train_samples)
        self.assertTrue(val_samples)
        self.assertEqual(sorted({sample.label for sample in train_samples}), [0, 1])
        self.assertEqual(sorted({sample.label for sample in val_samples}), [0, 1])

    def test_client_split_covers_all_samples(self):
        samples = [MedicalSample("s", idx, idx % 2, "nih") for idx in range(12)]
        clients = split_samples_to_clients(samples, num_clients=3, seed=11)
        flattened = [sample.item_index for client in clients for sample in client]
        self.assertEqual(sorted(flattened), list(range(12)))


if __name__ == "__main__":
    unittest.main()
