class ERPSpellerModelRLDA(components.Algorithm):
    """Classical model for ERP-based spellers using regularized linear
    discriminant analysis (rLDA)
    """
    def __init__(self):
        """Class constructor
        """
        # Super call
        super().__init__()

        # Methods
        prep = StandardPreprocessing()
        feat_ext = StandardFeatureExtraction()
        clf = components.ProcessingClassWrapper(
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
            fit=[], predict_proba=['y_pred']
        )
        feat_ext_connector = \
            components.ProcessingFuncWrapper(self.feat_ext_connector,
                                             outputs=['x', 'x_info'])
        cmd_decoding = components.ProcessingFuncWrapper(
            self.command_decoder,
            outputs=['spell_result', 'spell_result_per_seq']
        )
        assessment = components.ProcessingFuncWrapper(
            command_decoding_accuracy_per_seq,
            outputs=['spell_acc_per_seq']
        )

        # Add methods
        self.add_method('prep', prep)
        self.add_method('feat-ext', feat_ext)
        self.add_method('feat-ext-connector', feat_ext_connector)
        self.add_method('cmd-clf', clf)
        self.add_method('cmd-decoding', cmd_decoding)
        self.add_method('assessment', assessment)

        # Fit pipeline
        self.add_pipeline('fit-dataset', self.__fit_dataset_pipeline())
        self.add_pipeline('decode-commands', self.__decode_commands_pipeline())

    def __fit_dataset_pipeline(self):
        pipe = components.Pipeline()
        uid_0 = pipe.input(['dataset'])
        uid_1 = pipe.add(
            method_func_key='prep:transform_dataset',
            dataset=pipe.conn_to(uid_0, 'dataset')
        )
        uid_2 = pipe.add(
            method_func_key='feat-ext:transform_dataset',
            dataset=pipe.conn_to(uid_1, 'dataset'),
        )
        uid_3 = pipe.add(
            method_func_key='feat-ext-connector:feat_ext_connector',
            x=pipe.conn_to(uid_2, 'x'),
            x_info=pipe.conn_to(uid_2, 'x_info')
        )
        uid_4 = pipe.add(
            method_func_key='clf:fit',
            X=pipe.conn_to(uid_2, 'x'),
            y=pipe.conn_to(
                uid_3, 'x_info',
                conn_exp=lambda x_info: x_info['erp_labels']
            )
        )
        uid_5 = pipe.add(
            method_func_key='clf:predict_proba',
            X=pipe.conn_to(uid_2, 'x')
        )
        uid_6 = pipe.add(
            method_func_key='cmd-decoding:command_decoder',
            y_pred=pipe.conn_to(
                uid_5, 'y_pred',
                conn_exp=lambda y_pred: y_pred[:, 1]
            ),
            x_info=pipe.conn_to(uid_3, 'x_info'),
        )
        uid_7 = pipe.add(
            method_func_key='assessment:command_decoding_accuracy_per_seq',
            selected_commands_per_seq=pipe.conn_to(
                uid_6, 'spell_result_per_seq'
            ),
            target_commands=pipe.conn_to(
                uid_3, 'x_info',
                conn_exp=lambda x_info: x_info['spell_target']
            )
        )
        return pipe

    def __decode_commands_pipeline(self):
        # Decode command pipeline
        pipe = components.Pipeline()
        uid_0 = pipe.input(['times', 'signal', 'fs', 'x_info'])
        uid_1 = pipe.add(
            method_func_key='prep:transform_signal',
            signal=pipe.conn_to(uid_0, 'signal'),
            fs=pipe.conn_to(uid_0, 'fs')
        )
        uid_2 = pipe.add(
            method_func_key='feat-ext:transform_signal',
            times=pipe.conn_to(uid_0, 'times'),
            signal=pipe.conn_to(uid_1, 'signal'),
            fs=pipe.conn_to(uid_0, 'fs'),
            onsets=pipe.conn_to(
                uid_0, 'x_info',
                conn_exp=lambda x_info: x_info['onsets']
            ),
        )
        uid_4 = pipe.add(
            method_func_key='clf:predict_proba',
            X=pipe.conn_to(uid_2, 'x'),
        )
        uid_5 = pipe.add(
            method_func_key='cmd-decoding:command_decoder',
            y_pred=pipe.conn_to(
                uid_4, 'y_pred',
                conn_exp=lambda y_pred: y_pred[:, 1]
            ),
            x_info=pipe.conn_to(uid_0, 'x_info')
        )
        return pipe

    @staticmethod
    def feat_ext_connector(x, x_info):
        return x, x_info

    @staticmethod
    def command_decoder(y_pred, x_info):
        # Spell result per sequence
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        return spell_result, spell_result_per_seq

    def fit_dataset(self, dataset):
        return self.exec_pipeline('fit-dataset', dataset=dataset)

    def decode_commands(self, times, signal, fs, x_info):
        return self.exec_pipeline('decode-commands', times=times,
                                  signal=signal, fs=fs, x_info=x_info)


class ERPSpellerModelEEGInception(components.Algorithm):
    """Model for ERP-based spellers using EEG-Inception
    """
    def __init__(self, control_state_detection=False):
        """Class constructor
        """
        # Super call
        super().__init__()

        # Only import deep learning models if necessary
        from medusa import deep_learning_models

        # Variables
        self.control_state_detection = control_state_detection

        # 1. Preprocessing
        prep = StandardPreprocessing()
        self.add_method('prep', prep)

        # 2. Feature extraction
        feat_ext = StandardFeatureExtraction(
            target_fs=128,
            concatenate_channels=False
        )
        feat_ext_connector = \
            components.ProcessingFuncWrapper(self.feat_ext_connector,
                                             outputs=['x', 'x_info'])
        self.add_method('feat-ext', feat_ext)
        self.add_method('feat-ext-connector', feat_ext_connector)

        # 3. Command decoding methods
        cmd_clf = deep_learning_models.EEGInceptionv1()
        cmd_decoding = components.ProcessingFuncWrapper(
            self.command_decoder,
            outputs=['spell_result', 'spell_result_per_seq']
        )
        cmd_assessment = components.ProcessingFuncWrapper(
            command_decoding_accuracy_per_seq,
            outputs=['cmd_acc_per_seq']
        )
        self.add_method('cmd-clf', cmd_clf)
        self.add_method('cmd-decoding', cmd_decoding)
        self.add_method('cmd-assessment', cmd_assessment)

        # 4. Control state detection methods
        if self.control_state_detection:
            csd_clf = deep_learning_models.EEGInceptionv1()
            csd_decoding = components.ProcessingFuncWrapper(
                detect_control_state,
                outputs=['selected_control_state',
                         'selected_control_state_per_seq',
                         'state_scores']
            )
            csd_assessment = components.ProcessingFuncWrapper(
                control_state_detection_accuracy_per_seq,
                outputs=['csd_acc_per_seq']
            )
            self.add_method('csd-clf', csd_clf)
            self.add_method('csd-decoding', csd_decoding)
            self.add_method('csd-assessment', csd_assessment)

        # Add pipelines
        self.add_pipeline('fit-dataset', self.__fit_dataset_pipeline())
        self.add_pipeline('decode-commands', self.__decode_commands_pipeline())

    def __fit_dataset_pipeline(self):
        pipe = components.Pipeline()
        uid_0 = pipe.input(['dataset'])
        uid_1 = pipe.add(
            method_func_key='prep:transform_dataset',
            dataset=pipe.conn_to(uid_0, 'dataset')
        )
        uid_2 = pipe.add(
            method_func_key='feat-ext:transform_dataset',
            dataset=pipe.conn_to(uid_1, 'dataset'),
        )
        uid_3 = pipe.add(
            method_func_key='feat-ext-connector:feat_ext_connector',
            x=pipe.conn_to(uid_2, 'x'),
            x_info=pipe.conn_to(uid_2, 'x_info')
        )
        uid_4 = pipe.add(
            method_func_key='cmd-clf:fit',
            X=pipe.conn_to(uid_2, 'x'),
            y=pipe.conn_to(
                uid_3, 'x_info',
                conn_exp=lambda x_info: x_info['erp_labels']
            ),
            validation_split=0.2
        )
        uid_5 = pipe.add(
            method_func_key='clf:predict_proba',
            X=pipe.conn_to(uid_2, 'x')
        )
        uid_6 = pipe.add(
            method_func_key='cmd-decoding:command_decoder',
            y_pred=pipe.conn_to(
                uid_5, 'y_pred',
                conn_exp=lambda y_pred: y_pred[:, 1]
            ),
            x_info=pipe.conn_to(uid_3, 'x_info'),
        )
        uid_7 = pipe.add(
            method_func_key='cmd-assessment:command_decoding_accuracy_per_seq',
            selected_commands_per_seq=pipe.conn_to(
                uid_6, 'spell_result_per_seq'
            ),
            target_commands=pipe.conn_to(
                uid_3, 'x_info',
                conn_exp=lambda x_info: x_info['spell_target']
            )
        )
        if self.control_state_detection:
            uid_8 = pipe.add(
                method_func_key='csd-clf:fit',
                X=pipe.conn_to(uid_2, 'x'),
                y=pipe.conn_to(
                    uid_3, 'x_info',
                    conn_exp=lambda x_info: x_info['control_state_labels']
                ),
                validation_split=0.2
            )
            uid_9 = pipe.add(
                method_func_key='clf:predict_proba',
                X=pipe.conn_to(uid_2, 'x')
            )
            uid_10 = pipe.add(
                method_func_key='csd-decoding:detect_control_state',
                y_pred=pipe.conn_to(
                    uid_5, 'y_pred',
                    conn_exp=lambda y_pred: y_pred[:, 1]
                ),
                run_idx=pipe.conn_to(
                    uid_3, 'x_info',
                    conn_exp=lambda x_info: x_info['run_idx']
                ),
                trial_idx=pipe.conn_to(
                    uid_3, 'x_info',
                    conn_exp=lambda x_info: x_info['trial_idx']
                ),
                sequence_idx=pipe.conn_to(
                    uid_3, 'x_info',
                    conn_exp=lambda x_info: x_info['sequence_idx']
                ),
            )
            uid_11 = pipe.add(
                method_func_key='csd-assessment:control_state_'
                                'detection_accuracy_per_seq',
                selected_commands_per_seq=pipe.conn_to(
                    uid_6, 'spell_result_per_seq'
                ),
                target_commands=pipe.conn_to(
                    uid_3, 'x_info',
                    conn_exp=lambda x_info: x_info['spell_target']
                )
            )
        return pipe

    def __decode_commands_pipeline(self):
        # Decode command pipeline
        pipe = components.Pipeline()
        uid_0 = pipe.input(['times', 'signal', 'fs', 'x_info'])
        uid_1 = pipe.add(
            method_func_key='prep:transform_signal',
            signal=pipe.conn_to(uid_0, 'signal'),
            fs=pipe.conn_to(uid_0, 'fs')
        )
        uid_2 = pipe.add(
            method_func_key='feat-ext:transform_signal',
            times=pipe.conn_to(uid_0, 'times'),
            signal=pipe.conn_to(uid_1, 'signal'),
            fs=pipe.conn_to(uid_0, 'fs'),
            onsets=pipe.conn_to(
                uid_0, 'x_info',
                conn_exp=lambda x_info: x_info['onsets']
            ),
        )
        uid_4 = pipe.add(
            method_func_key='cmd-clf:predict_proba',
            X=pipe.conn_to(uid_2, 'x'),
        )
        uid_5 = pipe.add(
            method_func_key='cmd-decoding:command_decoder',
            y_pred=pipe.conn_to(
                uid_4, 'y_pred',
                conn_exp=lambda y_pred: y_pred[:, 1]
            ),
            x_info=pipe.conn_to(uid_0, 'x_info')
        )
        return pipe

    @staticmethod
    def feat_ext_connector(x, x_info):
        return x, x_info

    @staticmethod
    def command_decoder(y_pred, x_info):
        # Spell result per sequence
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        return spell_result, spell_result_per_seq

    def fit_dataset(self, dataset):
        return self.exec_pipeline('fit-dataset', dataset=dataset)

    def decode_commands(self, times, signal, fs, x_info):
        return self.exec_pipeline('decode-commands', times=times,
                                  signal=signal, fs=fs, x_info=x_info)
