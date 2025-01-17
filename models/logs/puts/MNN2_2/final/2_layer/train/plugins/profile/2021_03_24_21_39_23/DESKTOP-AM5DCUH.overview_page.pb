�	���	+:@���	+:@!���	+:@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���	+:@���R��6@1��p�q�?Aޓ��ZӬ?I�L�:i@*	������P@2U
Iterator::Model::ParallelMapV2jM�?!�ٳ�ь<@)jM�?1�ٳ�ь<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate8��d�`�?!�8\��=@)���_vO�?1��*�`6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�!��u��?!����5@)��0�*�?1�w�ge�1@:Preprocessing2F
Iterator::Model���S㥛?!oAV�-D@)vq�-�?1FR��	�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��ZӼ�t?!�s�,G~@)��ZӼ�t?1�s�,G~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY�8��m�?!����M@)�J�4q?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�� ��@)HP�s�b?1�� ��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!&�D�$@@)_�Q�[?1LM.s/T@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���X@Q����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���R��6@���R��6@!���R��6@      ��!       "	��p�q�?��p�q�?!��p�q�?*      ��!       2	ޓ��ZӬ?ޓ��ZӬ?!ޓ��ZӬ?:	�L�:i@�L�:i@!�L�:i@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���X@y����?�"C
%gradient_tape/sequential/dense/MatMulMatMul��*�2}�?!��*�2}�?0"5
sequential/dense/MatMulMatMul��*�2}�?!��*�2}�?0"7
sequential/dense_1/MatMulMatMulq2bS�?!7Yc��&�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulzA���R�?!���'h�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�r{P�P�?!M+.�F2�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�Q�����?!�U��8��?"7
sequential/dense_2/MatMulMatMul�Q�����?!��E*��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchzA���R�?!�M9���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradzA���R�?!!��,���?"E
'gradient_tape/sequential/dense_2/MatMulMatMulzA���R�?!(���?0Q      Y@Y>����/@aX�i��U@q�5�Y�X@yO�I�p�?"�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 