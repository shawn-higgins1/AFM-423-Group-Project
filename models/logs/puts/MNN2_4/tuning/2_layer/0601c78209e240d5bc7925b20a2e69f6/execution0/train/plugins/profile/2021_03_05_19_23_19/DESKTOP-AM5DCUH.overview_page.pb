�	�%Z�`6@�%Z�`6@!�%Z�`6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�%Z�`6@b��U��3@1��ao�?A0*��D�?I��q5�@*	�����H@2U
Iterator::Model::ParallelMapV2�������?!�_̀��9@)�������?1�_̀��9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!d�o�@F<@)g��j+��?11E>�S8@:Preprocessing2F
Iterator::Model+�����?!S�n0ED@)y�&1�|?1��6�$-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*�?!�����i5@)��0�*x?1���?�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n��r?!�$m�K"@)/n��r?1�$m�K"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!�L�ϺM@)y�&1�l?1��6�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!���Й@)ŏ1w-!_?1���Й@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!�����7@)a2U0*�S?1V?�[�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI+�(�X@Q�;5յ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	b��U��3@b��U��3@!b��U��3@      ��!       "	��ao�?��ao�?!��ao�?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	��q5�@��q5�@!��q5�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q+�(�X@y�;5յ�?�"5
sequential/dense/MatMulMatMul�Bt���?!�Bt���?0"C
%gradient_tape/sequential/dense/MatMulMatMul�E�P��?!D�{̲?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulbɃ�°�?!�(H�v��?"7
sequential/dense_1/MatMulMatMulbɃ�°�?!��l>�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulk�&{T��?!����?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul���?�?!@��\���?"7
sequential/dense_2/MatMulMatMulo��m��?! ƒ�?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile����ݱ�?!"=.���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchbɃ�°�?!N���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradbɃ�°�?!��,��?Q      Y@Y>����/@aX�i��U@q�k���zX@y��= 0��?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 