
C
sgemm_32x32x32_NN*28�O@�OH�OXbsequential/dense/MatMulh
Q
sgemm_32x32x32_TN*28�G@�GH�GXb%gradient_tape/sequential/dense/MatMulh
I
sgemm_32x32x32_NN_vec*28�@@�@H�@Xbsequential/dense_1/MatMulh
W
sgemm_32x32x32_NT_vec*28�@@�@H�@Xb'gradient_tape/sequential/dense_1/MatMulh
W
sgemm_32x32x32_TN_vec*28�?@�?H�?b)gradient_tape/sequential/dense_1/MatMul_1h
�
{_Z13gemv2N_kernelIiiffffLi128ELi32ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_*28�0@�0H�0b)gradient_tape/sequential/dense_2/MatMul_1h
�
�_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_*28�(@�(H�(Xbsequential/dense_2/MatMulh
�
�_ZN10tensorflow67_GLOBAL__N__43_dynamic_stitch_op_gpu_cu_compute_80_cpp1_ii_1a3d8a6f19DynamicStitchKernelIiEEviiNS_20GpuDeviceArrayStructIiLi8EEENS2_IPKT_Li8EEEPS4_*28�(@�(H�(b.gradient_tape/mean_squared_error/DynamicStitchh
�
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*28�(@�(H�(b4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradh
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE*28�@�H�Xb'gradient_tape/sequential/dense_2/MatMulh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b$Adam/Adam/update_2/ResourceApplyAdamh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b$Adam/Adam/update_4/ResourceApplyAdamh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b$Adam/Adam/update_5/ResourceApplyAdamh
�
t_ZN10tensorflow7functor17BlockReduceKernelIPdS2_Li256ENS0_3SumIdEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_7h
�
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*28�@�H�b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradh
�
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*28�@�H�b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b
LogicalAndh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b	truediv_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_18scalar_quotient_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b(gradient_tape/mean_squared_error/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/Powh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�btruedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bsubh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b5gradient_tape/mean_squared_error/weighted_loss/Tile_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKdLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_Casth
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b$Adam/Adam/update_3/ResourceApplyAdamh
(

Abs_kernel*28�@�H�bAbs_1h
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b$Adam/Adam/update_1/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bmul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bdiv_no_nan_1h
&

Abs_kernel*28�@�H�bAbsh
b
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28�@�H�bsequential/dense/BiasAddh
d
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28�@�H�bsequential/dense_1/BiasAddh
d
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28�@�H�bsequential/dense_2/BiasAddh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28�@�H�b"Adam/Adam/update/ResourceApplyAdamh
�
t_ZN10tensorflow7functor17BlockReduceKernelIPdS2_Li256ENS0_3SumIdEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_4h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPdS2_Li256ENS0_3SumIdEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_5h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPdS2_Li256ENS0_3SumIdEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_6h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_2h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�b$mean_squared_error/weighted_loss/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIddNS0_20scalar_difference_opIddEELb0EEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bsub_4h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bmul_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bmul_5h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b	truediv_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bsub_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_28scalar_squared_difference_opIdEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bsub_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorCwiseNullaryOpINS0_15scalar_const_opIdEEKS8_EEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bbroadcast_weights_1/ones_likeh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIdKNS4_INS5_IKfLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCast_5h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKdSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_6h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKdSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_8h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKdSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_9h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b$gradient_tape/mean_squared_error/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffLi1EEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bMaximumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b
div_no_nanh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bdiv_no_nan_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b&mean_squared_error/weighted_loss/valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b
Adam/Pow_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b&gradient_tape/mean_squared_error/mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_28scalar_squared_difference_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b$mean_squared_error/SquaredDifferenceh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS_19TensorCwiseBinaryOpINS0_13scalar_cmp_opIKfSC_LNS0_14ComparisonNameE5EEEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISC_EESH_EEEESH_KNS_18TensorCwiseUnaryOpINS0_10bind2nd_opINS0_17scalar_product_opISC_SC_EEEESH_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bBgradient_tape/sequential/dense/leaky_re_lu/LeakyRelu/LeakyReluGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS_19TensorCwiseBinaryOpINS0_13scalar_cmp_opIKfSC_LNS0_14ComparisonNameE5EEEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISC_EESH_EEEESH_KNS_18TensorCwiseUnaryOpINS0_10bind2nd_opINS0_17scalar_product_opISC_SC_EEEESH_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bFgradient_tape/sequential/dense_1/leaky_re_lu_1/LeakyRelu/LeakyReluGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS_19TensorCwiseBinaryOpINS0_13scalar_cmp_opIKfSC_LNS0_14ComparisonNameE5EEEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISC_EESH_EEEESH_KNS_18TensorCwiseUnaryOpINS0_10bind2nd_opINS0_17scalar_product_opISC_SC_EEEESH_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b&sequential/dense/leaky_re_lu/LeakyReluh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS_19TensorCwiseBinaryOpINS0_13scalar_cmp_opIKfSC_LNS0_14ComparisonNameE5EEEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISC_EESH_EEEESH_KNS_18TensorCwiseUnaryOpINS0_10bind2nd_opINS0_17scalar_product_opISC_SC_EEEESH_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b*sequential/dense_1/leaky_re_lu_1/LeakyReluh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKdLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bsequential/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCast_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCast_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b2mean_squared_error/weighted_loss/num_elements/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKxLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/Cast_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_4h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_5h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIxxEEKNS4_INS5_IKxLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/Adam/AssignAddVariableOph
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_10h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKdSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOp_7h
(

Abs_kernel*28�@�H�bAbs_2h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28�@�H�bSum_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAssignAddVariableOph
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bsub_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIddEEKNS4_INS5_IKdLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bmul_4h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIdKNS4_INS5_IKfLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_Cast_5h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIdLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIdKNS4_INS5_IKfLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCast_6h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b$gradient_tape/mean_squared_error/subh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bCast_4h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b%gradient_tape/mean_squared_error/Casth