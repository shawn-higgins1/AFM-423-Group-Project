�	W]�jJb7@W]�jJb7@!W]�jJb7@	^f H��?^f H��?!^f H��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6W]�jJb7@T����4@1��=�4�?A�+e�X�?I]3�f��@Y������`?*	33333�G@2U
Iterator::Model::ParallelMapV2�]K�=�?!tI,S<@)�]K�=�?1tI,S<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!���\�:@)46<�R�?1Sz��6@:Preprocessing2F
Iterator::ModelM�O��?!I�1�NE@)lxz�,C|?1:�g *-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!M����&6@)a��+ey?1�Y��)*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!���ɹ$"@)"��u��q?1���ɹ$"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!��O�%�L@)�����g?1�Kb��x@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���7a@)��H�}]?1���7a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�Kb��x8@)/n��R?1w���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9^f H��?I
3D3�X@Q��b_��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	T����4@T����4@!T����4@      ��!       "	��=�4�?��=�4�?!��=�4�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	]3�f��@]3�f��@!]3�f��@B      ��!       J	������`?������`?!������`?R      ��!       Z	������`?������`?!������`?b      ��!       JGPUY^f H��?b q
3D3�X@y��b_��?�"5
sequential/dense/MatMulMatMult���p��?!t���p��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�g��GY�?!�}Pd�n�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�g��GY�?!�1IW��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�g��GY�?!�� %��?"7
sequential/dense_1/MatMulMatMul%uf�=\�?!e�mߙ��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchd�mߙ��?!�y[�e�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMuld�mߙ��?!�1IW��?"7
sequential/dense_2/MatMulMatMuld�mߙ��?!��6�s��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�g��GY�?!������?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�g��GY�?!�Y���?Q      Y@Y>����/@aX�i��U@q/�f�aOW@y�1IW��?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 