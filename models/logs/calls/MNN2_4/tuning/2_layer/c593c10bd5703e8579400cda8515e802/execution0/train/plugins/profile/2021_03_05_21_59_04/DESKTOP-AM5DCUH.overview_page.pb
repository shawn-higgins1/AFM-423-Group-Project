�	���B��6@���B��6@!���B��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���B��6@N�q�:4@1�5�!��?Ag��j+��?I�n�Ʃ@*	����̌G@2U
Iterator::Model::ParallelMapV2�
F%u�?!����:@)�
F%u�?1����:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!x��t��9@)M�O��?1'����q5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!���-48@)� �	�?1�?T#Y0@:Preprocessing2F
Iterator::Model46<��?!���?mC@)Ǻ���v?1^̸4y�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!W��N)l@)���_vOn?1W��N)l@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��H�}�?!�O[h��N@)F%u�k?1�^���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!F����@)�J�4a?1F����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�]K�=�?!�r���<<@)ŏ1w-!_?1d���"@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|�J̭�X@Q!W�T�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N�q�:4@N�q�:4@!N�q�:4@      ��!       "	�5�!��?�5�!��?!�5�!��?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	�n�Ʃ@�n�Ʃ@!�n�Ʃ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q|�J̭�X@y!W�T�?�"5
sequential/dense/MatMulMatMul�^ш���?!�^ш���?0"7
sequential/dense_1/MatMulMatMul'!�`�٠?!���t0�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�p�W٠?!Jx�?�S�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�p�W٠?!PoD`�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�p�W٠?!{��Ꙗ�?"7
sequential/dense_2/MatMulMatMulͭ��?!5�M�8�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�L�{��?!������?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�-g���?!��Vfx�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�p�Wِ?!���H���?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�p�Wِ?!���^��?0Q      Y@Y>����/@aX�i��U@q�e�|^�W@y xژS�?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 